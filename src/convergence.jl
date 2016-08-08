using ProgressMeter

# Given an exact solution, test the convergence of the propagator, as
# a function of the amount of time steps.
function convergence(exact, y0, tmax, Na, Nb, propagator::MagnusPropagator)
    Ns = round(Int, logspace(Na, Nb, 100))
    errors = Vector{Float64}(length(Ns))
    times = Vector{Float64}(length(Ns))
    @showprogress for j = eachindex(Ns)
        tic()
        r = integrate(y0, tmax, Ns[j], propagator)
        times[j] = toq()
        errors[j] = sqrt(sumabs2(exact-r))
    end
    Ns,errors,times
end
