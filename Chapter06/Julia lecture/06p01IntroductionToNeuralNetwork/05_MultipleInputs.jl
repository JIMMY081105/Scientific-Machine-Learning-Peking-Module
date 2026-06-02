# =====================================================================================
# 6.1 - A network with multiple inputs
# =====================================================================================
# GOAL: extend a neuron to two inputs, NN(x1, x2) = tanh(W*[x1; x2] + b), and visualise
#       it as a contour plot over the (x1, x2) plane.
# =====================================================================================

using CairoMakie

x1range = -5:0.02:5
x2range = -5:0.02:5

W = [0.3 1.0]
b = [1]
function f(x)
    return tanh.(W * x + b)
end

x = [[x1i, x2i] for x1i in x1range, x2i in x2range]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"x_1", ylabel = L"x_2", title = "Neural Network", limits = (-5.0, 5.0, -5.0, 5.0))
contour!(ax, x1range, x2range, only.(f.(x)); labels = true, colormap = :hsv, linewidth = 5)
fig
