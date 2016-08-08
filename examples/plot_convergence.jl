function plot_convergence(exact, y0, tmax, Na, Nb, propagator::MagnusPropagator, subs, label)
    Ns,errors,times = Magnus.convergence(exact, y0, tmax, Na, Nb, propagator)
    subplot(subs[1])
    plot(Ns, errors, ".-", label = label)
    subplot(subs[2])
    plot(errors, times, ".-", label = label)
end
