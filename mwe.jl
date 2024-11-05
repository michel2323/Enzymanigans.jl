using Enzyme
# Required presently
#Enzyme.API.runtimeActivity!(true)
Enzyme.API.looseTypeAnalysis!(true)
Enzyme.API.maxtypeoffset!(2032)

mutable struct HydrostaticFreeSurfaceModel2{TS, 
                                           V} 

             advection :: V        # Advection scheme for tracers
           timestepper :: TS       # Object containing timestepper fields and parameters
end

function HydrostaticFreeSurfaceModel2()

    advection = nothing

    timestepper = nothing
    model = HydrostaticFreeSurfaceModel2(advection,
                                        timestepper)


    return model
end



f(grid) = CenterField(grid)

const maximum_diffusivity = 100

function time_step2!(model, Δt;
                    callbacks=[], euler=false)

    Δt == 0 && @warn "Δt == 0 may cause model blowup!"
end

function checkpoint_struct_for2(body::Function, scheme, model, range)
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
    alg,
    model::MT,
    shadowmodel::MT,
    range,
) where {MT}
    println("MWE hello")
    body(model)
    Enzyme.autodiff(Reverse, Const(body), Duplicated(model, shadowmodel))
end

function momentum_equation!(model, scheme)
    
    Δt = 1e-1# * Δz^2

    begin
        let
            if !(1:10 isa UnitRange{Int64})
                error("Checkpointing.jl: Only UnitRange{Int64} is supported.")
            end
            i = 1
            model = checkpoint_struct_for2(scheme, model, 1:10) do model
                begin
                    time_step2!(model, Δt; euler = true)
                end
                i += 1
                nothing
            end
        end
    end

    return zero(Float64)
end

model = HydrostaticFreeSurfaceModel2()
dmodel = Enzyme.make_zero(model)

revolve = nothing
momentum_equation!(model, revolve)


du²_dκ = autodiff(
    set_runtime_activity(Enzyme.Reverse),
    momentum_equation!,
    Duplicated(model, dmodel),
    Const(revolve)
)
