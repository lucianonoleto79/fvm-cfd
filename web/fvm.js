/*
 * fvm.js — Solver de Volumes Finitos para escoamento incompressível viscoso 2D
 * ---------------------------------------------------------------------------
 * Equações (adimensionais, rho = 1, U = L = 1, nu = 1/Re):
 *     div(u) = 0
 *     div(u u) = -grad(p) + (1/Re) lap(u)
 *
 * Método:
 *   - Volumes finitos em malha cartesiana uniforme DESENCONTRADA (staggered,
 *     Harlow & Welch 1965): p no centro das células, u nas faces verticais,
 *     v nas faces horizontais — evita o desacoplamento par-ímpar da pressão.
 *   - Convecção: upwind implícito + correção diferida (deferred correction,
 *     Khosla & Rubin 1974) para diferenças centrais -> 2ª ordem no regime
 *     convergido. gamma = 0 => upwind puro; gamma = 1 => CDS.
 *   - Difusão: diferenças centrais (2ª ordem).
 *   - Acoplamento pressão-velocidade: SIMPLEC (Van Doormaal & Raithby 1984).
 *   - Sistemas lineares: momento por TDMA linha-a-linha alternado (ADI);
 *     correção de pressão (simétrica, Neumann puro) por Gradiente Conjugado
 *     pré-condicionado (Jacobi).
 *
 * Problema implementado: cavidade com tampa deslizante (lid-driven cavity),
 * domínio [0,1]x[0,1], tampa superior com u = 1, demais paredes em repouso.
 *
 * Indexação (k = j*ni + i):
 *   p  : nx    x ny      células,   p(i,j)  em ((i+.5)dx, (j+.5)dy)
 *   u  : (nx+1) x ny     faces "x", u(i,j)  em (i dx,     (j+.5)dy)
 *   v  : nx    x (ny+1)  faces "y", v(i,j)  em ((i+.5)dx, j dy)
 */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------- álgebra
  /** Sistema de 5 pontos:  aP phi_P = aE phi_E + aW phi_W + aN phi_N + aS phi_S + b */
  function System(ni, nj) {
    const n = ni * nj;
    this.ni = ni; this.nj = nj;
    this.aP = new Float64Array(n); this.aE = new Float64Array(n);
    this.aW = new Float64Array(n); this.aN = new Float64Array(n);
    this.aS = new Float64Array(n); this.b = new Float64Array(n);
  }

  /** Resíduo L1 do sistema: sum |aP phi - sum anb phi_nb - b| */
  function residualL1(A, phi) {
    const { ni, nj, aP, aE, aW, aN, aS, b } = A;
    let r = 0;
    for (let j = 0; j < nj; j++) {
      for (let i = 0; i < ni; i++) {
        const k = j * ni + i;
        let s = b[k] - aP[k] * phi[k];
        if (i < ni - 1) s += aE[k] * phi[k + 1];
        if (i > 0) s += aW[k] * phi[k - 1];
        if (j < nj - 1) s += aN[k] * phi[k + ni];
        if (j > 0) s += aS[k] * phi[k - ni];
        r += Math.abs(s);
      }
    }
    return r;
  }

  /** Varreduras TDMA linha-a-linha alternadas (x depois y) — algoritmo de Thomas. */
  function solveADI(A, phi, sweeps, P, Q) {
    const { ni, nj, aP, aE, aW, aN, aS, b } = A;
    for (let s = 0; s < sweeps; s++) {
      // linhas em x (j fixo)
      for (let j = 0; j < nj; j++) {
        for (let i = 0; i < ni; i++) {
          const k = j * ni + i;
          let d = b[k];
          if (j < nj - 1) d += aN[k] * phi[k + ni];
          if (j > 0) d += aS[k] * phi[k - ni];
          const aw = i > 0 ? aW[k] : 0;
          const den = aP[k] - (i > 0 ? aw * P[i - 1] : 0);
          P[i] = (i < ni - 1 ? aE[k] : 0) / den;
          Q[i] = (d + (i > 0 ? aw * Q[i - 1] : 0)) / den;
        }
        for (let i = ni - 1; i >= 0; i--) {
          const k = j * ni + i;
          phi[k] = (i < ni - 1 ? P[i] * phi[k + 1] : 0) + Q[i];
        }
      }
      // linhas em y (i fixo)
      for (let i = 0; i < ni; i++) {
        for (let j = 0; j < nj; j++) {
          const k = j * ni + i;
          let d = b[k];
          if (i < ni - 1) d += aE[k] * phi[k + 1];
          if (i > 0) d += aW[k] * phi[k - 1];
          const as = j > 0 ? aS[k] : 0;
          const den = aP[k] - (j > 0 ? as * P[j - 1] : 0);
          P[j] = (j < nj - 1 ? aN[k] : 0) / den;
          Q[j] = (d + (j > 0 ? as * Q[j - 1] : 0)) / den;
        }
        for (let j = nj - 1; j >= 0; j--) {
          const k = j * ni + i;
          phi[k] = (j < nj - 1 ? P[j] * phi[k + ni] : 0) + Q[j];
        }
      }
    }
  }

  /** y = A x  (operador  aP x - sum anb x_nb) */
  function matvec(A, x, y) {
    const { ni, nj, aP, aE, aW, aN, aS } = A;
    for (let j = 0; j < nj; j++) {
      for (let i = 0; i < ni; i++) {
        const k = j * ni + i;
        let s = aP[k] * x[k];
        if (i < ni - 1) s -= aE[k] * x[k + 1];
        if (i > 0) s -= aW[k] * x[k - 1];
        if (j < nj - 1) s -= aN[k] * x[k + ni];
        if (j > 0) s -= aS[k] * x[k - ni];
        y[k] = s;
      }
    }
  }

  // ------------------------------------------------ multigrid (pré-condicionador)
  /**
   * Multigrid de correção aditiva (Hutchinson & Raithby 1986): blocos de 2x2
   * volumes são aglomerados; os coeficientes grossos são a soma dos
   * coeficientes finos que cruzam a fronteira do bloco (equivalente a Galerkin
   * com prolongamento constante por partes). Conservativo por construção.
   */
  function buildMG(A0) {
    const levels = [{ A: A0, r: new Float64Array(A0.ni * A0.nj) }];
    let ni = A0.ni, nj = A0.nj;
    while (ni % 2 === 0 && nj % 2 === 0 && ni > 4 && nj > 4) {
      ni /= 2; nj /= 2;
      const n = ni * nj;
      levels.push({ A: new System(ni, nj), x: new Float64Array(n), rhs: new Float64Array(n), r: new Float64Array(n) });
    }
    return levels;
  }

  function coarsenMG(levels) {
    for (let l = 1; l < levels.length; l++) {
      const F = levels[l - 1].A, C = levels[l].A;
      C.aP.fill(0); C.aE.fill(0); C.aW.fill(0); C.aN.fill(0); C.aS.fill(0);
      for (let j = 0; j < F.nj; j++) {
        for (let i = 0; i < F.ni; i++) {
          const k = j * F.ni + i, K = (j >> 1) * C.ni + (i >> 1);
          C.aP[K] += F.aP[k];
          if (i & 1) C.aE[K] += F.aE[k]; else C.aP[K] -= F.aE[k]; // vizinho interno ao bloco
          if (i & 1) C.aP[K] -= F.aW[k]; else C.aW[K] += F.aW[k];
          if (j & 1) C.aN[K] += F.aN[k]; else C.aP[K] -= F.aN[k];
          if (j & 1) C.aP[K] -= F.aS[k]; else C.aS[K] += F.aS[k];
        }
      }
    }
  }

  /** Gauss-Seidel ponto a ponto, varredura direta ou reversa. */
  function gaussSeidel(A, x, rhs, forward) {
    const { ni, nj, aP, aE, aW, aN, aS } = A, n = ni * nj;
    for (let c = 0; c < n; c++) {
      const k = forward ? c : n - 1 - c, i = k % ni, j = (k - i) / ni;
      let s = rhs[k];
      if (i < ni - 1) s += aE[k] * x[k + 1];
      if (i > 0) s += aW[k] * x[k - 1];
      if (j < nj - 1) s += aN[k] * x[k + ni];
      if (j > 0) s += aS[k] * x[k - ni];
      x[k] = s / aP[k];
    }
  }

  /** Ciclo V simétrico (GS direto antes, reverso depois) => pré-condicionador SPD. */
  function vcycle(levels, l, x, rhs) {
    const { A, r } = levels[l];
    x.fill(0);
    if (l === levels.length - 1) {
      for (let s = 0; s < 20; s++) { gaussSeidel(A, x, rhs, true); gaussSeidel(A, x, rhs, false); }
      return;
    }
    gaussSeidel(A, x, rhs, true); gaussSeidel(A, x, rhs, true);
    matvec(A, x, r);
    for (let k = 0; k < r.length; k++) r[k] = rhs[k] - r[k];
    const C = levels[l + 1], ni = A.ni, nic = C.A.ni;
    C.rhs.fill(0);
    for (let k = 0; k < r.length; k++) { const i = k % ni, j = (k - i) / ni; C.rhs[(j >> 1) * nic + (i >> 1)] += r[k]; }
    vcycle(levels, l + 1, C.x, C.rhs);
    for (let k = 0; k < x.length; k++) { const i = k % ni, j = (k - i) / ni; x[k] += C.x[(j >> 1) * nic + (i >> 1)]; }
    gaussSeidel(A, x, rhs, false); gaussSeidel(A, x, rhs, false);
  }

  /**
   * Gradiente Conjugado pré-condicionado (multigrid, ou Jacobi se mg = null)
   * para a equação de correção de pressão (simétrica, semidefinida). Como só
   * há condições de Neumann, remove-se a média do lado direito
   * (compatibilidade) e da solução.
   */
  function solvePCG(A, x, tol, maxIt, w, mg) {
    const n = x.length, b = A.b, aP = A.aP;
    const { r, z, p, q } = w;
    const precond = mg ? () => vcycle(mg, 0, z, r) : () => { for (let k = 0; k < n; k++) z[k] = r[k] / aP[k]; };
    let mean = 0;
    for (let k = 0; k < n; k++) mean += b[k];
    mean /= n;
    let bnorm = 0;
    for (let k = 0; k < n; k++) { r[k] = b[k] - mean; bnorm += r[k] * r[k]; x[k] = 0; }
    bnorm = Math.sqrt(bnorm);
    if (bnorm === 0) return 0;
    let rz = 0;
    precond();
    for (let k = 0; k < n; k++) { p[k] = z[k]; rz += r[k] * z[k]; }
    let it = 0;
    for (; it < maxIt; it++) {
      matvec(A, p, q);
      let pq = 0;
      for (let k = 0; k < n; k++) pq += p[k] * q[k];
      const alpha = rz / pq;
      let rr = 0;
      for (let k = 0; k < n; k++) { x[k] += alpha * p[k]; r[k] -= alpha * q[k]; rr += r[k] * r[k]; }
      if (Math.sqrt(rr) < tol * bnorm) { it++; break; }
      let rzNew = 0;
      precond();
      for (let k = 0; k < n; k++) rzNew += r[k] * z[k];
      const beta = rzNew / rz;
      rz = rzNew;
      for (let k = 0; k < n; k++) p[k] = z[k] + beta * p[k];
    }
    mean = 0;
    for (let k = 0; k < n; k++) mean += x[k];
    mean /= n;
    for (let k = 0; k < n; k++) x[k] -= mean;
    return it;
  }

  // ---------------------------------------------------------------- solver
  function CavitySolver(opts) {
    opts = opts || {};
    this.Re = opts.Re || 100;
    this.nx = opts.N || 128;
    this.ny = opts.N || 128;
    if (this.nx % 2) throw new Error('N deve ser par (linha central sobre faces)');
    this.Ulid = 1;
    this.alphaU = opts.alphaU != null ? opts.alphaU : 0.8; // sub-relaxação do momento
    this.alphaP = opts.alphaP != null ? opts.alphaP : 1.0; // SIMPLEC dispensa relaxar p
    this.gamma = opts.gamma != null ? opts.gamma : 1.0;    // 1 = CDS via correção diferida
    this.adiSweeps = opts.adiSweeps || 2;
    this.pcgTol = opts.pcgTol || 1e-3;
    this.tol = opts.tol || 1e-6;

    const nx = this.nx, ny = this.ny;
    this.dx = 1 / nx; this.dy = 1 / ny;
    this.u = new Float64Array((nx + 1) * ny);
    this.v = new Float64Array(nx * (ny + 1));
    this.p = new Float64Array(nx * ny);
    this.pc = new Float64Array(nx * ny);
    this.dU = new Float64Array((nx + 1) * ny);
    this.dV = new Float64Array(nx * (ny + 1));
    this.Au = new System(nx + 1, ny);
    this.Av = new System(nx, ny + 1);
    this.Ap = new System(nx, ny);
    this.mg = opts.multigrid === false ? null : buildMG(this.Ap);
    const m = Math.max(nx, ny) + 2;
    this.P = new Float64Array(m); this.Q = new Float64Array(m);
    const n = nx * ny;
    this.work = { r: new Float64Array(n), z: new Float64Array(n), p: new Float64Array(n), q: new Float64Array(n) };
    this.iter = 0;
    this.history = []; // {it, ru, rv, rm}
    this.converged = false;
  }

  /** Monta a equação de u (faces verticais). Retorna o resíduo normalizado. */
  CavitySolver.prototype.assembleU = function () {
    const { nx, ny, dx, dy, u, v, p, Au, dU, alphaU, gamma, Ulid } = this;
    const nu = 1 / this.Re;
    const De = nu * dy / dx, Dn = nu * dx / dy;
    const ni = nx + 1;
    const { aP, aE, aW, aN, aS, b } = Au;
    let norm = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i <= nx; i++) {
        const k = j * ni + i;
        if (i === 0 || i === nx) { // parede lateral: u = 0 (Dirichlet no próprio nó)
          aP[k] = 1; aE[k] = aW[k] = aN[k] = aS[k] = 0; b[k] = 0; dU[k] = 0;
          continue;
        }
        // fluxos de massa nas faces do VC de u (interpolação linear)
        const Fe = 0.5 * (u[k] + u[k + 1]) * dy;
        const Fw = 0.5 * (u[k - 1] + u[k]) * dy;
        const Fn = 0.5 * (v[(j + 1) * nx + i - 1] + v[(j + 1) * nx + i]) * dx;
        const Fs = 0.5 * (v[j * nx + i - 1] + v[j * nx + i]) * dx;
        const uP = u[k], uE = u[k + 1], uW = u[k - 1];
        let ae = De + Math.max(-Fe, 0);
        let aw = De + Math.max(Fw, 0);
        let an, as, ap0 = 0, src = 0, dc = 0;
        // correção diferida: F*(phi_CDS - phi_UDS) na face, para o termo fonte
        dc += Fe * (0.5 * (uP + uE) - (Fe > 0 ? uP : uE));
        dc -= Fw * (0.5 * (uW + uP) - (Fw > 0 ? uW : uP));
        if (j === ny - 1) { // tampa: u = Ulid a dy/2 do nó
          an = 0; ap0 += 2 * Dn; src += 2 * Dn * Ulid;
        } else {
          an = Dn + Math.max(-Fn, 0);
          const uN = u[k + ni];
          dc += Fn * (0.5 * (uP + uN) - (Fn > 0 ? uP : uN));
        }
        if (j === 0) { // fundo: u = 0
          as = 0; ap0 += 2 * Dn;
        } else {
          as = Dn + Math.max(Fs, 0);
          const uS = u[k - ni];
          dc -= Fs * (0.5 * (uS + uP) - (Fs > 0 ? uS : uP));
        }
        const sumNb = ae + aw + an + as;
        ap0 += sumNb + (Fe - Fw + Fn - Fs);
        src += (p[j * nx + i - 1] - p[j * nx + i]) * dy - gamma * dc;
        aE[k] = ae; aW[k] = aw; aN[k] = an; aS[k] = as;
        aP[k] = ap0 / alphaU;
        b[k] = src + (1 - alphaU) * aP[k] * uP;
        const den = aP[k] - sumNb; // SIMPLEC
        dU[k] = dy / Math.max(den, 1e-3 * aP[k]);
        norm += Math.abs(ap0 * uP);
      }
    }
    return residualL1(Au, u) / Math.max(norm, 1e-30);
  };

  /** Monta a equação de v (faces horizontais). */
  CavitySolver.prototype.assembleV = function () {
    const { nx, ny, dx, dy, u, v, p, Av, dV, alphaU, gamma } = this;
    const nu = 1 / this.Re;
    const De = nu * dy / dx, Dn = nu * dx / dy;
    const nu1 = nx + 1;
    const { aP, aE, aW, aN, aS, b } = Av;
    let norm = 0;
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (j === 0 || j === ny) { // fundo/tampa: v = 0
          aP[k] = 1; aE[k] = aW[k] = aN[k] = aS[k] = 0; b[k] = 0; dV[k] = 0;
          continue;
        }
        const Fe = 0.5 * (u[(j - 1) * nu1 + i + 1] + u[j * nu1 + i + 1]) * dy;
        const Fw = 0.5 * (u[(j - 1) * nu1 + i] + u[j * nu1 + i]) * dy;
        const Fn = 0.5 * (v[k] + v[k + nx]) * dx;
        const Fs = 0.5 * (v[k - nx] + v[k]) * dx;
        const vP = v[k], vN = v[k + nx], vS = v[k - nx];
        let an = Dn + Math.max(-Fn, 0);
        let as = Dn + Math.max(Fs, 0);
        let ae, aw, ap0 = 0, src = 0, dc = 0;
        dc += Fn * (0.5 * (vP + vN) - (Fn > 0 ? vP : vN));
        dc -= Fs * (0.5 * (vS + vP) - (Fs > 0 ? vS : vP));
        if (i === nx - 1) { ae = 0; ap0 += 2 * De; }
        else {
          ae = De + Math.max(-Fe, 0);
          const vE = v[k + 1];
          dc += Fe * (0.5 * (vP + vE) - (Fe > 0 ? vP : vE));
        }
        if (i === 0) { aw = 0; ap0 += 2 * De; }
        else {
          aw = De + Math.max(Fw, 0);
          const vW = v[k - 1];
          dc -= Fw * (0.5 * (vW + vP) - (Fw > 0 ? vW : vP));
        }
        const sumNb = ae + aw + an + as;
        ap0 += sumNb + (Fe - Fw + Fn - Fs);
        src += (p[(j - 1) * nx + i] - p[j * nx + i]) * dx - gamma * dc;
        aE[k] = ae; aW[k] = aw; aN[k] = an; aS[k] = as;
        aP[k] = ap0 / alphaU;
        b[k] = src + (1 - alphaU) * aP[k] * vP;
        const den = aP[k] - sumNb;
        dV[k] = dx / Math.max(den, 1e-3 * aP[k]);
        norm += Math.abs(ap0 * vP);
      }
    }
    return residualL1(Av, v) / Math.max(norm, 1e-30);
  };

  /** Monta a equação de correção de pressão p'. Retorna o desbalanço de massa (L1). */
  CavitySolver.prototype.assembleP = function () {
    const { nx, ny, dx, dy, u, v, dU, dV, Ap } = this;
    const nu1 = nx + 1;
    const { aP, aE, aW, aN, aS, b } = Ap;
    let mass = 0;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        const ku = j * nu1 + i;
        aE[k] = i < nx - 1 ? dU[ku + 1] * dy : 0;
        aW[k] = i > 0 ? dU[ku] * dy : 0;
        aN[k] = j < ny - 1 ? dV[k + nx] * dx : 0;
        aS[k] = j > 0 ? dV[k] * dx : 0;
        aP[k] = aE[k] + aW[k] + aN[k] + aS[k];
        b[k] = (u[ku] - u[ku + 1]) * dy + (v[k] - v[k + nx]) * dx;
        mass += Math.abs(b[k]);
      }
    }
    return mass;
  };

  /** Uma iteração SIMPLEC. */
  CavitySolver.prototype.iterate = function () {
    const { nx, ny, u, v, p, pc, dU, dV } = this;
    const nu1 = nx + 1;
    // 1) momento (previsor) com o campo de pressão atual
    const ru = this.assembleU();
    solveADI(this.Au, u, this.adiSweeps, this.P, this.Q);
    const rv = this.assembleV();
    solveADI(this.Av, v, this.adiSweeps, this.P, this.Q);
    // 2) correção de pressão
    const rm = this.assembleP();
    if (this.mg) coarsenMG(this.mg);
    this.lastPcgIt = solvePCG(this.Ap, pc, this.pcgTol, 2000, this.work, this.mg);
    // 3) correções
    for (let j = 0; j < ny; j++)
      for (let i = 1; i < nx; i++) {
        const k = j * nu1 + i;
        u[k] += dU[k] * (pc[j * nx + i - 1] - pc[j * nx + i]);
      }
    for (let j = 1; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        v[k] += dV[k] * (pc[(j - 1) * nx + i] - pc[j * nx + i]);
      }
    let mean = 0;
    for (let k = 0; k < p.length; k++) { p[k] += this.alphaP * pc[k]; mean += p[k]; }
    mean /= p.length;
    for (let k = 0; k < p.length; k++) p[k] -= mean; // nível de referência: média nula
    this.iter++;
    const h = { it: this.iter, ru, rv, rm };
    this.history.push(h);
    if (this.iter > 10 && Math.max(ru, rv, rm) < this.tol) this.converged = true;
    return h;
  };

  CavitySolver.prototype.run = function (nIter) {
    let h = null;
    for (let n = 0; n < nIter && !this.converged; n++) h = this.iterate();
    return h;
  };

  // -------------------------------------------------------- pós-processamento
  /** u(y) na linha vertical x = 0.5 (inclui paredes y=0 e y=1). */
  CavitySolver.prototype.uCenterline = function () {
    const { nx, ny, dy, u } = this;
    const i = nx / 2, y = [0], val = [0];
    for (let j = 0; j < ny; j++) { y.push((j + 0.5) * dy); val.push(u[j * (nx + 1) + i]); }
    y.push(1); val.push(this.Ulid);
    return { y, u: val };
  };

  /** v(x) na linha horizontal y = 0.5 (inclui paredes). */
  CavitySolver.prototype.vCenterline = function () {
    const { nx, ny, dx, v } = this;
    const j = ny / 2, x = [0], val = [0];
    for (let i = 0; i < nx; i++) { x.push((i + 0.5) * dx); val.push(v[j * nx + i]); }
    x.push(1); val.push(0);
    return { x, v: val };
  };

  /** Função de corrente nos vértices: psi = integral de u dy a partir do fundo. */
  CavitySolver.prototype.streamFunction = function () {
    const { nx, ny, dy, u } = this;
    const ni = nx + 1, psi = new Float64Array(ni * (ny + 1));
    for (let i = 0; i <= nx; i++)
      for (let j = 0; j < ny; j++)
        psi[(j + 1) * ni + i] = psi[j * ni + i] + u[j * ni + i] * dy;
    return { psi, ni, nj: ny + 1, dx: this.dx, dy };
  };

  /** Centro do vórtice primário: mínimo de psi com refino parabólico sub-malha. */
  CavitySolver.prototype.primaryVortex = function () {
    const { psi, ni, nj, dx, dy } = this.streamFunction();
    let kmin = 0;
    for (let k = 0; k < psi.length; k++) if (psi[k] < psi[kmin]) kmin = k;
    const i = kmin % ni, j = (kmin / ni) | 0;
    const par = (fm, f0, fp) => { const d = fm - 2 * f0 + fp; return d !== 0 ? 0.5 * (fm - fp) / d : 0; };
    let sx = 0, sy = 0;
    if (i > 0 && i < ni - 1) sx = par(psi[kmin - 1], psi[kmin], psi[kmin + 1]);
    if (j > 0 && j < nj - 1) sy = par(psi[kmin - ni], psi[kmin], psi[kmin + ni]);
    return { psi: psi[kmin], x: (i + sx) * dx, y: (j + sy) * dy };
  };

  /** Interpolação linear de (xs, ys) em xq. */
  function interp(xs, ys, xq) {
    if (xq <= xs[0]) return ys[0];
    for (let k = 1; k < xs.length; k++)
      if (xq <= xs[k]) { const t = (xq - xs[k - 1]) / (xs[k] - xs[k - 1]); return ys[k - 1] + t * (ys[k] - ys[k - 1]); }
    return ys[ys.length - 1];
  }

  const api = { CavitySolver, System, solveADI, solvePCG, interp };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FVM = api;
})(typeof self !== 'undefined' ? self : this);
