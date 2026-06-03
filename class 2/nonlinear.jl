using CairoMakie   # plotting package

# ---------- Circuit data ----------
# Five resistances (Ohms) and two source voltages (Volts)
R₁ = 5
R₂ = 150
R₃ = 100
R₄ = 250
R₅ = 200
V₁ = 100.0
V₂ = 50.0

# ---------- System of equations ----------
# The circuit gives 5 linear equations for the 5 unknown branch
# currents i₁ … i₅ (Kirchhoff's voltage and current laws).
# In matrix form:  A * i = C₁
# Each ROW of A is one equation; each COLUMN belongs to one current.
# Rows 1–3: voltage loops (coefficients are resistances),
# rows 4–5: current balance at the nodes (coefficients ±1).
A = [R₁  R₂   0   0   0;
      0 -R₂  R₃  R₄   0;
      0   0   0  R₄ -R₅;
      1  -1  -1   0   0;
      0   0   1  -1  -1]

# Right-hand side: the known voltages (one entry per equation)
C₁ = [V₁, 0, V₂, 0, 0]

# Solving A*i = C₁ for the currents: the backslash operator "\"
# means "solve the linear system" (like inv(A)*C₁ but better).
i = A \ C₁          # i is a vector: i[1]=i₁, i[2]=i₂, …, i[5]=i₅

# ---------- Parameter sweep ----------
# Question: how does the current i₅ change when the source V₁
# is varied from 50 V to 100 V (in steps of 1 V)?
V1vec = 50:1:100                 # all the V₁ values to try
i5vec = zeros(length(V1vec))     # empty storage for the answers

for n in 1:length(V1vec)         # loop over every V₁ value
    V₁ = V1vec[n]                # pick the n-th voltage
    C₁ = [V₁, 0, V₂, 0, 0]       # rebuild the right-hand side with it
    i = A \ C₁                   # solve the system again
    i5vec[n] = i[5]              # keep only the 5th current
end

# ---------- Plot: i₅ as a function of V₁ ----------
fig = Figure()
ax = Axis(fig[1, 1],
    xlabel = L"V_1 \; [V]",      # L"..." makes nice math labels
    ylabel = L"i_5 \; [A]",
    title  = "Current i₅ vs source voltage V₁")
lines!(ax, V1vec, i5vec, color = :blue)
display(fig)                     # show the plot in VS Code's plot pane
