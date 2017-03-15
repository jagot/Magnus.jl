import Base.LinAlg.BLAS: @blasfunc
import Base.LinAlg: BlasInt
import Base.LinAlg.LAPACK: stegr!
const liblapack = Base.liblapack_name

type stegr_work{T<:Number}
    jobz::Char
    range::Char
    dv::Vector{T}
    ev::Vector{T}
    vl::Real
    vu::Real
    il::BlasInt
    iu::BlasInt
    abstol::Vector{T}
    m::Vector{BlasInt}
    w::Vector{T}
    Z::Matrix{T}
    isuppz::Vector{BlasInt}
    work::Vector{T}
    lwork::BlasInt
    iwork::Vector{BlasInt}
    liwork::BlasInt
    info::Vector{BlasInt}
end
for (stegr,elty) in ((:dstegr_,:Float64),
                     (:sstegr_,:Float32))
    @eval begin
        function stegr!(n::BlasInt, sw::stegr_work{$elty})
            ldz = stride(sw.Z, 2)
            ccall((@blasfunc($stegr), liblapack), Void,
                  (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  &sw.jobz, &sw.range, &n,
                  sw.dv, sw.ev,
                  &sw.vl, &sw.vu, &sw.il, &sw.iu,
                  sw.abstol, sw.m,
                  sw.w, sw.Z, &ldz,
                  sw.isuppz, sw.work, &sw.lwork, sw.iwork, &sw.liwork,
                  sw.info)
        end
    end
end

function stegr_work(T::DataType, n::BlasInt,
                    jobz::Char = 'V', range::Char = 'A')
    dv = Array(T, n)
    ev = Array(T, n)
    abstol = Array(T, 1)
    m = Vector{BlasInt}(1)
    w = Array(T, n)
    ldz = jobz == 'N' ? 1 : n
    Z = Array(T, ldz, n)
    isuppz = Array{BlasInt}(2n)
    work = Array(T, 1)
    lwork = -one(BlasInt)
    iwork = Array{BlasInt}(1)
    liwork = -one(BlasInt)
    info = Array(BlasInt, 1)
    sw = stegr_work(jobz, range,
                    dv, ev,
                    0.0, 0.0,
                    BlasInt(0), BlasInt(0),
                    abstol, m,
                    w, Z,
                    isuppz,
                    work, lwork,
                    iwork, liwork,
                    info)
    stegr!(n, sw)
    sw.lwork = BlasInt(sw.work[1])
    sw.work = Array(T, sw.lwork)
    sw.liwork = sw.iwork[1]
    sw.iwork = Array(BlasInt, sw.liwork)
    sw
end

function expT{T<:Number, R<:Real}(α::AbstractVector{R}, β::AbstractVector{R},
                                  τ::T, v::AbstractVector{T},
                                  sw::stegr_work{R})
    copy!(sw.dv, α)
    copy!(sw.ev, β)
    n = BlasInt(length(α))
    stegr!(n, sw)
    for i = 1:n
        v[i] = exp(τ*sw.w[i])*sw.Z[1,i]
    end
    v[:] = sw.Z[1:n,1:n]*v
end
