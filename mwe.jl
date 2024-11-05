using Enzyme
import .EnzymeRules: augmented_primal, reverse, Annotation, has_rrule_from_sig
using .EnzymeRules

mutable struct Model
end

function bar!()
    @warn "segfaulting soon"
    # Doesn't sefault with a println
    # println("segfaulting soon")
    return nothing
end

function loop(body::Function, model)
    body(model)
    return model
end

function augmented_primal(config, func::Const{typeof(loop)}, ret, body, model)
    func.val(body.val, model.val)
    return AugmentedReturn(nothing, nothing, nothing)
end

function reverse(config, ::Const{typeof(loop)}, dret::Type{<:Const}, tape, body, model::Duplicated,)
    rev_loop(body.val, model.val, model.dval)
    return (nothing, nothing)
end

function rev_loop(body::Function, model::Model, shadowmodel::Model) 
    body(model)
    Enzyme.autodiff(Reverse, Const(body), Duplicated(model, shadowmodel))
end

function foo(model)
    model = loop(model) do model
            bar!()
    end
    return nothing
end

model = Model()
dmodel = Enzyme.make_zero(model)
foo(model)
autodiff(Enzyme.Reverse, foo, Duplicated(model, dmodel))
