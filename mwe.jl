using Enzyme
# Required presently
#Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

using Oceananigans
using Oceananigans.TurbulenceClosures: with_tracers
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField
using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames, validate_model_halo,
    validate_momentum_advection, validate_tracer_advection, validate_biogeochemistry,
    validate_closure, validate_velocity_boundary_conditions, validate_free_surface,
    validate_buoyancy, tupleit, biogeochemical_auxiliary_fields,
    adapt_advection_order, regularize_buoyancy, extract_boundary_conditions,
    regularize_field_boundary_conditions, add_closure_specific_boundary_conditions,
    HydrostaticFreeSurfaceVelocityFields, HydrostaticFreeSurfaceTendencyFields,
    TracerFields, PressureField, DiffusivityFields, materialize_free_surface,
    implicit_diffusion_solver, time_discretization, model_forcing, update_state!,
    TimeStepper, hydrostatic_prognostic_fields, set!
using Oceananigans.Fields: FunctionField
using Oceananigans: architecture
using KernelAbstractions
# import Checkpointing: Revolve, @checkpoint_struct
using Checkpointing 
Periodic = Oceananigans.Periodic

mutable struct HydrostaticFreeSurfaceModel2{TS, E, A<:Oceananigans.AbstractArchitecture, S,
                                           G, T, V, B, R, F, P, BGC, U, C, Φ, K, AF} <: Oceananigans.Models.AbstractModel{TS}

          architecture :: A        # Computer `Architecture` on which `Model` is run
                  grid :: G        # Grid of physical points on which `Model` is solved
                 clock :: Clock{T} # Tracks iteration number and simulation time of `Model`
             advection :: V        # Advection scheme for tracers
              buoyancy :: B        # Set of parameters for buoyancy model
              coriolis :: R        # Set of parameters for the background rotation rate of `Model`
          free_surface :: S        # Free surface parameters and fields
               forcing :: F        # Container for forcing functions defined by the user
               closure :: E        # Diffusive 'turbulence closure' for all model fields
             particles :: P        # Particle set for Lagrangian tracking
       biogeochemistry :: BGC      # Biogeochemistry for Oceananigans tracers
            velocities :: U        # Container for velocity fields `u`, `v`, and `w`
               tracers :: C        # Container for tracer fields
              pressure :: Φ        # Container for hydrostatic pressure
    diffusivity_fields :: K        # Container for turbulent diffusivities
           timestepper :: TS       # Object containing timestepper fields and parameters
      auxiliary_fields :: AF       # User-specified auxiliary fields for forcing functions and boundary conditions
end

function HydrostaticFreeSurfaceModel2(; grid,
                                     clock = Clock{eltype(grid)}(time = 0),
                                     momentum_advection = VectorInvariant(),
                                     tracer_advection = CenteredSecondOrder(),
                                     buoyancy = nothing,
                                     coriolis = nothing,
                                     free_surface = default_free_surface(grid, gravitational_acceleration=g_Earth),
                                     tracers = nothing,
                                     forcing::NamedTuple = NamedTuple(),
                                     closure = nothing,
                                     boundary_conditions::NamedTuple = NamedTuple(),
                                     particles::Oceananigans.Models.HydrostaticFreeSurfaceModels.ParticlesOrNothing = nothing,
                                     biogeochemistry::Oceananigans.Models.HydrostaticFreeSurfaceModels.AbstractBGCOrNothing = nothing,
                                     velocities = nothing,
                                     pressure = nothing,
                                     diffusivity_fields = nothing,
                                     auxiliary_fields = NamedTuple())

    # Check halos and throw an error if the grid's halo is too small
    @apply_regionally validate_model_halo(grid, momentum_advection, tracer_advection, closure)

    # Validate biogeochemistry (add biogeochemical tracers automagically)
    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)
    biogeochemical_fields = merge(auxiliary_fields, biogeochemical_auxiliary_fields(biogeochemistry))
    tracers, auxiliary_fields = validate_biogeochemistry(tracers, biogeochemical_fields, biogeochemistry, grid, clock)

    # Reduce the advection order in directions that do not have enough grid points
    @apply_regionally momentum_advection = validate_momentum_advection(momentum_advection, grid)
    default_tracer_advection, tracer_advection = validate_tracer_advection(tracer_advection, grid)
    default_generator(name, tracer_advection) = default_tracer_advection

    # Generate tracer advection scheme for each tracer
    tracer_advection_tuple = with_tracers(tracernames(tracers), tracer_advection, default_generator, with_velocities=false)
    momentum_advection_tuple = (; momentum = momentum_advection)
    advection = merge(momentum_advection_tuple, tracer_advection_tuple)
    advection = NamedTuple(name => adapt_advection_order(scheme, grid) for (name, scheme) in pairs(advection))

    validate_buoyancy(buoyancy, tracernames(tracers))
    buoyancy = regularize_buoyancy(buoyancy)

    # Collect boundary conditions for all model prognostic fields and, if specified, some model
    # auxiliary fields. Boundary conditions are "regularized" based on the _name_ of the field:
    # boundary conditions on u, v are regularized assuming they represent momentum at appropriate
    # staggered locations. All other fields are regularized assuming they are tracers.
    # Note that we do not regularize boundary conditions contained in *tupled* diffusivity fields right now.
    #
    # First, we extract boundary conditions that are embedded within any _user-specified_ field tuples:
    embedded_boundary_conditions = merge(extract_boundary_conditions(velocities),
                                         extract_boundary_conditions(tracers),
                                         extract_boundary_conditions(pressure),
                                         extract_boundary_conditions(diffusivity_fields))

    # Next, we form a list of default boundary conditions:
    prognostic_field_names = (:u, :v, :w, tracernames(tracers)..., :η, keys(auxiliary_fields)...)
    default_boundary_conditions = NamedTuple{prognostic_field_names}(Tuple(FieldBoundaryConditions()
                                                                           for name in prognostic_field_names))

    # Then we merge specified, embedded, and default boundary conditions. Specified boundary conditions
    # have precedence, followed by embedded, followed by default.
    boundary_conditions = merge(default_boundary_conditions, embedded_boundary_conditions, boundary_conditions)
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, prognostic_field_names)

    # Finally, we ensure that closure-specific boundary conditions, such as
    # those required by CATKEVerticalDiffusivity, are enforced:
    boundary_conditions = add_closure_specific_boundary_conditions(closure,
                                                                   boundary_conditions,
                                                                   grid,
                                                                   tracernames(tracers),
                                                                   buoyancy)

    # Ensure `closure` describes all tracers
    closure = with_tracers(tracernames(tracers), closure)

    # Put CATKE first in the list of closures
    closure = validate_closure(closure)

    # Either check grid-correctness, or construct tuples of fields
    velocities         = HydrostaticFreeSurfaceVelocityFields(velocities, grid, clock, boundary_conditions)
    tracers            = TracerFields(tracers, grid, boundary_conditions)
    pressure           = PressureField(grid)
    diffusivity_fields = DiffusivityFields(diffusivity_fields, grid, tracernames(tracers), boundary_conditions, closure)

    @apply_regionally validate_velocity_boundary_conditions(grid, velocities)

    arch = architecture(grid)
    free_surface = validate_free_surface(arch, free_surface)
    free_surface = materialize_free_surface(free_surface, velocities, grid)

    # Instantiate timestepper if not already instantiated
    implicit_solver = implicit_diffusion_solver(time_discretization(closure), grid)
    timestepper = TimeStepper(:QuasiAdamsBashforth2, grid, tracernames(tracers);
                              implicit_solver = implicit_solver,
                              Gⁿ = HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, grid, tracernames(tracers)),
                              G⁻ = HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, grid, tracernames(tracers)))

    # Regularize forcing for model tracer and velocity fields.
    model_fields = merge(hydrostatic_prognostic_fields(velocities, free_surface, tracers), auxiliary_fields)
    forcing = model_forcing(model_fields; forcing...)
    
    model = HydrostaticFreeSurfaceModel2(arch, grid, clock, advection, buoyancy, coriolis,
                                        free_surface, forcing, closure, particles, biogeochemistry, velocities, tracers,
                                        pressure, diffusivity_fields, timestepper, auxiliary_fields)

    # update_state!(model; compute_tendencies = false)

    return model
end


#EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true

f(grid) = CenterField(grid)

const maximum_diffusivity = 100

function time_step2!(model, Δt;
                    callbacks=[], euler=false)

    Δt == 0 && @warn "Δt == 0 may cause model blowup!"
end

function checkpoint_struct_for2(body::Function, scheme::Scheme, model, range)
    for gensym() in range
        body(model)
    end
    return model
end

using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

function augmented_primal(
    config,
    func::Const{typeof(checkpoint_struct_for2)},
    ret,
    body,
    alg,
    model,
    range,
)
    tape_model = deepcopy(model.val)
    func.val(body.val, alg.val, model.val, range.val)
    if needs_primal(config)
        return AugmentedReturn(nothing, nothing, (tape_model,))
    else
        return AugmentedReturn(nothing, nothing, (tape_model,))
    end
end

function reverse(
    config,
    ::Const{typeof(checkpoint_struct_for2)},
    dret::Type{<:Const},
    tape,
    body,
    alg,
    model::Duplicated,
    range,
)
    (model_input,) = tape
    rev_checkpoint_struct_for2(
    # Checkpointing.rev_checkpoint_struct_for(
        body.val,
        alg.val,
        model_input,
        model.dval,
        range.val,
    )
    return (nothing, nothing, nothing, nothing)
end

function rev_checkpoint_struct_for2(
    body::Function,
    alg::Revolve,
    model::MT,
    shadowmodel::MT,
    range,
) where {MT}
    println("MWE hello")
    # model = deepcopy(model_input)
    # model_final = []
    body(model)
    # model_final = deepcopy(model)
    Enzyme.autodiff(Reverse, Const(body), Duplicated(model, shadowmodel))
    # return model_final
end

function momentum_equation!(model, scheme)
    
    # Do time-stepping
    Nx, Ny, Nz = size(model.grid)
    Δz = 1 / Nz
    Δt = 1e-1 * Δz^2

    model.clock.time = 0
    model.clock.iteration = 0

    # Loop to be checkpointed for AD
    # @macroexpand @checkpoint_struct scheme model for i = 1:10
    # # for i = 1:10
    #     time_step2!(model, Δt; euler=true)
    # end
    begin
    #= /disk/mschanen/julia_depot/packages/Checkpointing/RmOhf/src/Checkpointing.jl:171 =#
        let
            #= /disk/mschanen/julia_depot/packages/Checkpointing/RmOhf/src/Checkpointing.jl:172 =#
            if !(1:10 isa UnitRange{Int64})
                #= /disk/mschanen/julia_depot/packages/Checkpointing/RmOhf/src/Checkpointing.jl:173 =#
                error("Checkpointing.jl: Only UnitRange{Int64} is supported.")
            end
            #= /disk/mschanen/julia_depot/packages/Checkpointing/RmOhf/src/Checkpointing.jl:175 =#
            i = 1
            #= /disk/mschanen/julia_depot/packages/Checkpointing/RmOhf/src/Checkpointing.jl:176 =#
            model = checkpoint_struct_for2(scheme, model, 1:10) do model
            # model = Checkpointing.checkpoint_struct_for(scheme, model, 1:10) do model
                #= /disk/mschanen/julia_depot/packages/Checkpointing/RmOhf/src/Checkpointing.jl:181 =#
                begin
                    #= /disk/mschanen/git/Enzymanigans.jl/stable_diffusion/momentum_equation.jl:38 =#
                    time_step2!(model, Δt; euler = true)
                    #= /disk/mschanen/git/Enzymanigans.jl/stable_diffusion/momentum_equation.jl:39 =#
                end
                #= /disk/mschanen/julia_depot/packages/Checkpointing/RmOhf/src/Checkpointing.jl:182 =#
                i += 1
                #= /disk/mschanen/julia_depot/packages/Checkpointing/RmOhf/src/Checkpointing.jl:183 =#
                nothing
            end
        end
    end

    # Compute scalar metric
    u = model.velocities.u

    # Hard way (for enzyme - the sum function sometimes errors with AD)
    # c² = c^2
    # sum_c² = sum(c²)

    # Another way to compute it
    sum_u² = 0.0
    for k = 1:Nz, j = 1:Ny,  i = 1:Nx
        sum_u² += u[i, j, k]^2
    end

    # Need the ::Float64 for type inference with automatic differentiation
    return sum_u²::Float64
end

Nx = Ny = 32
Nz = 4

Lx = Ly = L = 2π
Lz = 1

x = y = (-L/2, L/2)
z = (-Lz, 0)
topology = (Periodic, Periodic, Bounded)

grid = RectilinearGrid(size=(Nx, Ny, Nz); x, y, z, topology)

u = XFaceField(grid)
v = YFaceField(grid)

U = 1
u₀(x, y, z) = - U * cos(x + L/8) * sin(y) * (-z)
v₀(x, y, z) = + U * sin(x + L/8) * cos(y) * (-z)

set!(u, u₀)
set!(v, v₀)
fill_halo_regions!(u)
fill_halo_regions!(v)

# TODO:
# 1. Make the velocity fields evolve
# 2. Add surface fluxes
# 3. Do a problem where we invert for the tracer fluxes (maybe with CATKE)

# buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState())

model = HydrostaticFreeSurfaceModel2(; grid,
                                    momentum_advection = WENO(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=1),
                                    #buoyancy = buoyancy,
                                    #tracers = (:T, :S),
                                    #velocities = PrescribedVelocityFields(; u, v),
                                    closure = nothing) #ScalarBiharmonicDiffusivity())

#set!(model, S = 34.7, T = 0.5)
# set!(model, u=u₀, v=v₀)

#=
# Compute derivative by hand
κ₁, κ₂ = 0.9, 1.1
u²₁ = momentum_equation!(model, 1, κ₁)
u²₂ = momentum_equation!(model, 1, κ₂)
du²_dκ_fd = (u²₂ - u²₁) / (κ₂ - κ₁)
=#

# Now for real
dmodel = Enzyme.make_zero(model)

u_old = model.velocities.u[:]
@show model.velocities.u
@show dmodel.velocities.u

# Revolve(#timesteps, #snapshots;...)
revolve = Revolve{typeof(model)}(
    10, 2; 
    verbose=1, 
    gc=true, 
    write_checkpoints=false
)
momentum_equation!(model, revolve)

revolve = Revolve{typeof(model)}(
    10, 2; 
    verbose=1, 
    gc=true, 
    write_checkpoints=false
)

du²_dκ = autodiff(
    set_runtime_activity(Enzyme.Reverse),
    momentum_equation!,
    Duplicated(model, dmodel),
    Const(revolve)
)

u_new = model.velocities.u[:]
@show model.velocities.u
@show dmodel.velocities.u

#@show u_new - u_old

#=
@info """ \n
Momentum Equation:
Enzyme computed $du²_dκ
Finite differences computed $du²_dκ_fd
"""
=#