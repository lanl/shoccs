We solve many similar systems with different boundary conditions, and cut-cell domains.  Rather than defining a new type with many similarities, can we instead define everything needed via expression templates (perhaps via constexpr), maybe the result could look something like

```c++
auto solver = build_solver<du/dt = div(u)>
auto u = solver(dt, u0, boundary_conditions, operators, )
```

Or, taking a page from `constexpr all the things`
```c++
auto solve = "du/dt = u.div(u)"_solver
```

Or what about
```c++
constexpr field_var u;
constexpr time_var t;
constexpr auto solver = build_solver(d(u)/d(t) = u.div(u))
```

