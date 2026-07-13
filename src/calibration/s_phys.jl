# =============================================================================
#  s_phys.jl — §2: S_phys — L2 cone-integrated physical sensor signal
#
#  S_phys(rh, κ_eff) is the ratio of the PMS5003-detected scattering signal
#  at humidity rh to the signal at the dry reference rh_dry:
#
#    S_phys(rh, κ) =
#      Σ_j w_j · D_wet(j)² · ∫_cone (i₁+i₂)(x_wet(j), n_dry, k_dry, μ) dμ
#      ──────────────────────────────────────────────────────────────────────
#      Σ_j w_j · D_dry(j)² · ∫_cone (i₁+i₂)(x_dry(j), n_dry, k_dry, μ) dμ
#
#  where:
#    w_j        — lognormal PSD quadrature weights (n(D)/D, log-spaced)
#    D_wet(j)   — wet diameter at PSD node j: D_dry(j) · GF(rh, κ)
#    x_wet(j)   — size parameter π·D_wet/λ
#    cone       — μ ∈ [cos(θ_max), cos(θ_min)] (PMS5003 acceptance cone)
#
#  S_phys ≡ 1 at rh = rh_dry by construction.  Replaces the power-law f(RH).
#
#  USAGE
#  -----
#    design = PMS5003_DEFAULT
#    cache  = build_s_phys_cache(design)
#    table  = build_s_phys_table(l2_emu, design, cache; n_rh=40, n_kappa=40)
#    val    = interp_s_phys(table, rh, kappa)         # hot-path lookup
#
#  Requires: angular_emulator.jl, physics.jl (growth_factor)
# =============================================================================

using LinearAlgebra, Printf, Statistics


# ---------------------------------------------------------------------------
# Design constants (CONST — not fittable; un-fittable by type system)
# ---------------------------------------------------------------------------

"""
    PMS5003Design

Hardware constants for the PMS5003 optical particle counter.
All fields are fixed at construction time and MUST NOT be included in any
optimisation parameter vector — they are physical design constants, not
instrument drift terms.

Geometry from teardown measurements; λ from LED spec sheet.
"""
struct PMS5003Design
    λ         :: Float32    # wavelength [µm]  — red LED 660 nm
    θ_min_deg :: Float32    # inner cone half-angle [deg]
    θ_max_deg :: Float32    # outer cone half-angle [deg]
    μ_min     :: Float32    # cos(θ_max_deg) — lower integration limit
    μ_max     :: Float32    # cos(θ_min_deg) — upper integration limit
    Dg        :: Float32    # lognormal geometric mean diameter [µm]
    σg        :: Float32    # geometric standard deviation
    n_dry     :: Float32    # dry real refractive index
    k_dry     :: Float32    # dry imaginary RI
    rh_dry    :: Float32    # reference dry RH [fraction]
    w_thresh  :: Float32    # wind speed flow-gate threshold [m/s]
    n_psd     :: Int        # PSD Gauss–Laguerre quadrature points
    n_cone    :: Int        # cone (μ) Gauss–Legendre quadrature points
end

function PMS5003Design(;
    λ         = 0.660f0,
    θ_min_deg = 30.0f0,
    θ_max_deg = 60.0f0,
    Dg        = 0.15f0,
    σg        = 1.60f0,
    n_dry     = 1.50f0,
    k_dry     = 0.00f0,
    rh_dry    = 0.20f0,
    w_thresh  = 5.0f0,
    n_psd     = 32,
    n_cone    = 20,
)
    μ_min = cos(deg2rad(Float32(θ_max_deg)))   # cos(60°) = 0.5
    μ_max = cos(deg2rad(Float32(θ_min_deg)))   # cos(30°) ≈ 0.866
    return PMS5003Design(λ, θ_min_deg, θ_max_deg, μ_min, μ_max,
                         Dg, σg, n_dry, k_dry, rh_dry, w_thresh, n_psd, n_cone)
end

const PMS5003_DEFAULT = PMS5003Design()


# ---------------------------------------------------------------------------
# Precomputed quadrature cache (built once; not differentiable targets)
# ---------------------------------------------------------------------------

struct SPHYSCache
    D_dry    :: Vector{Float32}    # PSD quadrature diameter nodes [µm]
    w_psd    :: Vector{Float32}    # PSD quadrature weights (sum=1)
    μ_cone   :: Vector{Float32}    # cone GL quadrature nodes on [μ_min, μ_max]
    w_cone   :: Vector{Float32}    # cone GL quadrature weights (sum=μ_max−μ_min)
    # Flattened batches for vectorised forward calls (n_psd × n_cone)
    n_rep    :: Vector{Float32}    # n_dry repeated (length n_psd*n_cone)
    k_rep    :: Vector{Float32}    # k_dry repeated
    μ_rep    :: Vector{Float32}    # μ_cone tiled over PSD points
end


"""
    gauss_legendre_unit(n) → (nodes, weights) on [-1,1]

Jacobi-eigenvalue GL quadrature for small n (no external packages).
Accurate to machine precision for n ≤ ~100.
"""
function gauss_legendre_unit(n::Int)
    β  = [0.5 / sqrt(1.0 - (2k)^(-2)) for k in 1:n-1]
    J  = SymTridiagonal(zeros(n), β)
    λv, V = eigen(J)
    w  = 2.0 .* V[1,:].^2
    return λv, w
end


"""
    build_s_phys_cache(design) → SPHYSCache

Precompute PSD and cone quadrature nodes and weights.
Call this once before building the S_phys table.
"""
function build_s_phys_cache(design::PMS5003Design = PMS5003_DEFAULT)
    # ── PSD: log-spaced nodes on [D_min, D_max] with lognormal weights ──────
    D_min = 0.02f0; D_max = 3.00f0
    n_psd = design.n_psd
    D_dry = Float32.(exp.(range(log(Float64(D_min)), log(Float64(D_max)), n_psd)))
    logσ  = log(Float32(design.σg))
    w_psd = exp.(-0.5f0 .* (log.(D_dry ./ design.Dg) ./ logσ).^2) ./ D_dry
    w_psd ./= sum(w_psd)

    # ── Cone: GL nodes on [μ_min, μ_max] via linear map from [-1,1] ─────────
    n_cone = design.n_cone
    ξ, w_gl = gauss_legendre_unit(n_cone)                     # nodes/weights on [-1,1]
    a = Float64(design.μ_min); b = Float64(design.μ_max)
    μ_cone  = Float32.((b - a) .* ξ ./ 2 .+ (a + b) / 2)
    w_cone  = Float32.((b - a) .* w_gl ./ 2)

    # ── Batch arrays for vectorised angular_forward_zyg ─────────────────────
    N_dc  = n_psd * n_cone
    n_rep = fill(design.n_dry, N_dc)
    k_rep = fill(design.k_dry, N_dc)
    # For PSD point j (1-indexed) and cone point k: index = (j-1)*n_cone + k
    # x_batch[...] = x_j  repeated for each k  → use inner repeat on x_all
    # μ_batch[...] = μ_k  repeated for each j  → tile n_psd times
    μ_rep = repeat(μ_cone, n_psd)   # [μ₁,μ₂,..., μ₁,μ₂,...] (n_psd blocks)

    return SPHYSCache(D_dry, w_psd, μ_cone, w_cone, n_rep, k_rep, μ_rep)
end


# ---------------------------------------------------------------------------
# S_phys direct computation (slow — for validation and table build)
# ---------------------------------------------------------------------------

"""
    s_phys_scalar(rh, κ_eff, l2_emu, design, cache) → Float64

Compute S_phys by direct L2 angular integration.  Slow (one MLP-batch per call)
but exact for the given l2_emu.  Use `build_s_phys_table` + `interp_s_phys`
for the ODE hot path.
"""
function s_phys_scalar(rh::Real, κ_eff::Real,
                        l2_emu::AngularEmulatorWeights,
                        design::PMS5003Design, cache::SPHYSCache)
    GF_wet = growth_factor(Float64(rh),         Float64(κ_eff))
    GF_dry = growth_factor(Float64(design.rh_dry), Float64(κ_eff))
    λ      = Float64(design.λ)

    D_dry  = Float64.(cache.D_dry)
    w_psd  = Float64.(cache.w_psd)
    μ_c    = Float64.(cache.μ_cone)
    w_c    = Float64.(cache.w_cone)
    n_psd  = length(D_dry)
    n_cone = length(μ_c)

    # Batch: n_psd × n_cone angular evaluations per (rh, κ) pair
    x_wet_all = Float32.((π / λ) .* D_dry .* GF_wet)   # (n_psd,)
    x_dry_all = Float32.((π / λ) .* D_dry .* GF_dry)

    # Repeat patterns: PSD index cycles slowest
    x_wet_batch = repeat(x_wet_all, inner=n_cone)
    x_dry_batch = repeat(x_dry_all, inner=n_cone)
    μ_batch     = Float32.(repeat(μ_c, n_psd))          # (n_psd*n_cone,)
    n_batch     = cache.n_rep
    k_batch     = cache.k_rep

    i1_wet, i2_wet = angular_forward_zyg(l2_emu, x_wet_batch, n_batch, k_batch, μ_batch)
    i1_dry, i2_dry = angular_forward_zyg(l2_emu, x_dry_batch, n_batch, k_batch, μ_batch)

    # Reshape to (n_psd, n_cone) and integrate over cone
    I_wet = reshape(Float64.(i1_wet .+ i2_wet), n_psd, n_cone)
    I_dry = reshape(Float64.(i1_dry .+ i2_dry), n_psd, n_cone)
    cone_int_wet = I_wet * w_c     # (n_psd,)  cone integral per PSD node
    cone_int_dry = I_dry * w_c

    D_wet_sq = D_dry .^ 2 .* GF_wet^2
    D_dry_sq = D_dry .^ 2 .* GF_dry^2

    num = sum(w_psd .* D_wet_sq .* cone_int_wet)
    den = sum(w_psd .* D_dry_sq .* cone_int_dry)
    return num / den
end


# ---------------------------------------------------------------------------
# Precomputed S_phys table  (analogous to MieFRHTable in mie_physics.jl)
# ---------------------------------------------------------------------------

struct SPHYSTable
    rh_grid    :: Vector{Float64}
    kappa_grid :: Vector{Float64}
    s_vals     :: Matrix{Float64}   # size (n_rh, n_kappa)
end


"""
    build_s_phys_table(l2_emu, design, cache; n_rh, n_kappa, ...) → SPHYSTable

Evaluate s_phys_scalar on a regular (rh, κ) grid.
Takes ~1–3 min for default 40×40 grid (1600 batch angular calls of size n_psd×n_cone).
"""
function build_s_phys_table(l2_emu::AngularEmulatorWeights,
                              design::PMS5003Design = PMS5003_DEFAULT,
                              cache::SPHYSCache = build_s_phys_cache(design);
                              n_rh   = 40,
                              n_kappa= 40,
                              rh_lo  = 0.15,
                              rh_hi  = 0.97,
                              k_lo   = 0.01,
                              k_hi   = 0.60,
                              verbose= true)
    rh_grid    = collect(range(rh_lo, rh_hi, n_rh))
    kappa_grid = collect(range(k_lo, k_hi, n_kappa))
    s_vals     = Matrix{Float64}(undef, n_rh, n_kappa)

    t0 = time()
    for (jk, κ) in enumerate(kappa_grid)
        for (ir, rh) in enumerate(rh_grid)
            s_vals[ir, jk] = s_phys_scalar(rh, κ, l2_emu, design, cache)
        end
        if verbose && mod(jk, max(1, n_kappa ÷ 5)) == 0
            @printf "  build_s_phys_table: %d/%d κ rows done (%.0fs)\n" jk n_kappa (time()-t0)
        end
    end
    return SPHYSTable(rh_grid, kappa_grid, s_vals)
end


"""
    interp_s_phys(table, rh, kappa) → S::Real

Bilinear interpolation into the precomputed S_phys table.
Clamps inputs to table bounds.  Zygote-differentiable w.r.t. kappa.
"""
function interp_s_phys(tbl::SPHYSTable, rh::Real, kappa::Real)
    rh_lo = tbl.rh_grid[1];    rh_hi = tbl.rh_grid[end]
    k_lo  = tbl.kappa_grid[1]; k_hi  = tbl.kappa_grid[end]
    n_rh  = length(tbl.rh_grid); n_k = length(tbl.kappa_grid)

    Δrh = (rh_hi - rh_lo) / (n_rh - 1)
    Δk  = (k_hi  - k_lo)  / (n_k  - 1)

    rh_c = clamp(rh,    rh_lo, rh_hi)
    k_c  = clamp(kappa, k_lo,  k_hi)

    ir_f = (rh_c - rh_lo) / Δrh
    ik_f = (k_c  - k_lo)  / Δk

    ir   = clamp(floor(Int, ir_f) + 1, 1, n_rh - 1)
    ik   = clamp(floor(Int, ik_f) + 1, 1, n_k  - 1)

    αr   = ir_f - floor(ir_f)
    αk   = ik_f - floor(ik_f)

    f00  = tbl.s_vals[ir,   ik  ]
    f10  = tbl.s_vals[ir+1, ik  ]
    f01  = tbl.s_vals[ir,   ik+1]
    f11  = tbl.s_vals[ir+1, ik+1]

    return (1-αr)*(1-αk)*f00 + αr*(1-αk)*f10 + (1-αr)*αk*f01 + αr*αk*f11
end
