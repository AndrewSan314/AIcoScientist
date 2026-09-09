import React, { useEffect, useRef, useState, useCallback } from 'react';
import { 
  RotateCw, 
  Maximize2, 
  Minimize2, 
  Zap,
  Crosshair
} from 'lucide-react';
import type { Candidate, CampaignStep } from '../types/mission_control';

interface StarkHologramSphereProps {
  candidates: Candidate[];
  selectedCandidateId?: string;
  onSelectCandidate?: (candidateId: string) => void;
  currentStep?: CampaignStep;
  isScanning?: boolean;
  onScanComplete?: (winnerId: string) => void;
}

interface AtomNode {
  id: string;
  label: string;
  composition: string;
  x0: number;
  y0: number;
  z0: number;
  x: number;
  y: number;
  z: number;
  sx: number;
  sy: number;
  scale: number;
  radius: number;
  color: string;
  glowColor: string;
  isWinner: boolean;
  isPareto: boolean;
  isTested: boolean;
  score: number;
  electronAngle: number;
}

export const StarkHologramSphere: React.FC<StarkHologramSphereProps> = ({
  candidates = [],
  selectedCandidateId,
  onSelectCandidate,
  currentStep,
  isScanning = false,
  onScanComplete,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [scanSequenceActive, setScanSequenceActive] = useState(false);
  const [scanStatusText, setScanStatusText] = useState('LATTICE READY // 1,035 CANDIDATES');
  const [hoveredAtom, setHoveredAtom] = useState<AtomNode | null>(null);
  const winnerId = currentStep?.preregistration?.action?.candidate_id || 'controlled-0';

  const [activeTargetId, setActiveTargetId] = useState<string>(
    selectedCandidateId || winnerId
  );

  const rotationRef = useRef({ x: 0.3, y: 0.6, z: 0.1 });
  const velocityRef = useRef({ x: 0.003, y: 0.007 });
  const isDraggingRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });
  const animFrameIdRef = useRef<number>(0);
  const atomsRef = useRef<AtomNode[]>([]);
  const scanProgressRef = useRef(0);

  useEffect(() => {
    if (selectedCandidateId) {
      setActiveTargetId(selectedCandidateId);
    } else if (winnerId) {
      setActiveTargetId(winnerId);
    }
  }, [selectedCandidateId, winnerId]);

  // Generate 3D Fibonacci Sphere Atom Lattice in Emerald & White
  useEffect(() => {
    const TOTAL_ATOMS = Math.max(160, Math.min(300, (candidates.length || 12) * 14));
    const goldenRatio = (1 + Math.sqrt(5)) / 2;
    const angleIncrement = Math.PI * 2 * goldenRatio;

    const realList: Candidate[] = candidates.length > 0 ? candidates : [
      { candidate_id: 'PG_0309', composition_label: 'LiCoO2 Precursor (0.33/0.33)', x: 0.82, y: 0.65, characterization_cost: 1.0, outcome_cost: 2.0 },
      { candidate_id: 'PG_0214', composition_label: 'Na0.67MnO2 High-Purity', x: 0.54, y: 0.42, characterization_cost: 1.0, outcome_cost: 2.0 },
      { candidate_id: 'PG_0182', composition_label: 'LiNi0.5Mn1.5O4 Spinel', x: 0.31, y: 0.78, characterization_cost: 1.0, outcome_cost: 2.0 },
      { candidate_id: 'controlled-0', composition_label: 'Syn-0 (Li-Mn Oxide)', x: 0.20, y: 0.55, characterization_cost: 1.0, outcome_cost: 2.0 },
      { candidate_id: 'controlled-1', composition_label: 'Syn-1 (Li-Co-Mn Oxide)', x: 0.45, y: 0.60, characterization_cost: 1.0, outcome_cost: 2.0 },
      { candidate_id: 'controlled-2', composition_label: 'Syn-2 (High VoI Pareto)', x: 0.75, y: 0.88, characterization_cost: 1.0, outcome_cost: 2.0 },
    ];

    const newAtoms: AtomNode[] = [];

    for (let i = 0; i < TOTAL_ATOMS; i++) {
      const t = i / TOTAL_ATOMS;
      const inclination = Math.acos(1 - 2 * t);
      const azimuth = angleIncrement * i;

      const x0 = Math.sin(inclination) * Math.cos(azimuth);
      const y0 = Math.sin(inclination) * Math.sin(azimuth);
      const z0 = Math.cos(inclination);

      const candidateMatch = realList[i % realList.length];
      const atomId = i < realList.length ? candidateMatch.candidate_id : `lattice-node-${i}`;
      const isWinner = atomId === activeTargetId || atomId === winnerId;
      const isPareto = i % 7 === 0 || isWinner;
      const isTested = i % 5 === 0;

      // STRICT EMERALD & WHITE THEME PALETTE
      let color = '#a7f3d0'; // Mint (untested candidate)
      let glowColor = 'rgba(167, 243, 208, 0.3)';

      if (isWinner) {
        color = '#ffffff'; // Pure brilliant white target
        glowColor = 'rgba(16, 185, 129, 0.95)';
      } else if (isPareto) {
        color = '#10b981'; // Vivid Emerald Green
        glowColor = 'rgba(16, 185, 129, 0.6)';
      } else if (isTested) {
        color = '#059669'; // Deep Forest Emerald
        glowColor = 'rgba(5, 150, 105, 0.5)';
      }

      newAtoms.push({
        id: atomId,
        label: atomId.replace('controlled-', 'Syn-'),
        composition: candidateMatch?.composition_label || `Atomic Formulation #${i + 1}`,
        x0,
        y0,
        z0,
        x: x0,
        y: y0,
        z: z0,
        sx: 0,
        sy: 0,
        scale: 1,
        radius: isWinner ? 5.5 : isPareto ? 4.0 : 2.8,
        color,
        glowColor,
        isWinner,
        isPareto,
        isTested,
        score: isWinner ? 0.482 : isPareto ? 0.35 + (i % 10) * 0.01 : 0.12 + (i % 10) * 0.01,
        electronAngle: Math.random() * Math.PI * 2,
      });
    }

    atomsRef.current = newAtoms;
  }, [candidates, activeTargetId, winnerId]);

  const triggerStarkScanSequence = useCallback(() => {
    setScanSequenceActive(true);
    setScanStatusText('// SCANNING 1,035 CANDIDATES: EVALUATING VoI //');
    scanProgressRef.current = 0;

    velocityRef.current = { x: 0.03, y: 0.07 };

    const startTime = Date.now();
    const DURATION = 2600;

    const scanInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(1, elapsed / DURATION);
      scanProgressRef.current = progress;

      if (progress < 0.45) {
        setScanStatusText(`// PHASE I: EMERALD LASER SWEEP [${Math.round(progress * 220)}%] //`);
      } else if (progress < 0.8) {
        setScanStatusText('// PHASE II: CONVERGING HIG + ΔU - C //');
        velocityRef.current.x *= 0.94;
        velocityRef.current.y *= 0.94;
      } else if (progress < 0.95) {
        setScanStatusText('// PHASE III: ALIGNING OPTIMAL VECTOR //');
      } else {
        clearInterval(scanInterval);
        velocityRef.current = { x: 0.002, y: 0.005 };
        setScanStatusText(`// TARGET LOCKED: ${activeTargetId} [VoI: +0.482] //`);
        setScanSequenceActive(false);
        if (onScanComplete) {
          onScanComplete(activeTargetId);
        }
      }
    }, 50);

    return () => clearInterval(scanInterval);
  }, [activeTargetId, onScanComplete]);

  useEffect(() => {
    if (isScanning && !scanSequenceActive) {
      triggerStarkScanSequence();
    }
  }, [isScanning, scanSequenceActive, triggerStarkScanSequence]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = canvas.parentElement?.clientWidth || 500);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 450);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', handleResize);

    let time = 0;

    const drawHolographicRings = (
      c: CanvasRenderingContext2D,
      cx: number,
      cy: number,
      r: number,
      rx: number,
      ry: number,
      isBack: boolean
    ) => {
      c.save();
      c.strokeStyle = isBack ? 'rgba(16, 185, 129, 0.07)' : 'rgba(16, 185, 129, 0.25)';
      c.lineWidth = 1.0;
      c.setLineDash([6, 8]);

      // Equator Ring
      c.beginPath();
      c.ellipse(cx, cy, r, r * Math.abs(Math.sin(rx)), ry, 0, Math.PI * 2);
      c.stroke();

      // Polar Meridian Ring
      c.strokeStyle = isBack ? 'rgba(5, 150, 105, 0.05)' : 'rgba(16, 185, 129, 0.18)';
      c.beginPath();
      c.ellipse(cx, cy, r * Math.abs(Math.cos(ry)), r, rx, 0, Math.PI * 2);
      c.stroke();

      if (!isBack) {
        c.setLineDash([]);
        c.strokeStyle = 'rgba(52, 211, 153, 0.35)';
        c.lineWidth = 1.0;
        for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 12) {
          const x1 = cx + Math.cos(angle) * r;
          const y1 = cy + Math.sin(angle) * r;
          const x2 = cx + Math.cos(angle) * (r - 5);
          const y2 = cy + Math.sin(angle) * (r - 5);
          c.beginPath();
          c.moveTo(x1, y1);
          c.lineTo(x2, y2);
          c.stroke();
        }
      }
      c.restore();
    };

    const drawStarkTargetLock = (
      c: CanvasRenderingContext2D,
      atom: AtomNode,
      cx: number,
      cy: number,
      t: number
    ) => {
      const { sx, sy } = atom;
      const lockSize = 24;

      c.save();
      // Rotating emerald dashed ring
      c.strokeStyle = 'rgba(16, 185, 129, 0.95)';
      c.lineWidth = 1.5;
      c.setLineDash([5, 5]);
      c.beginPath();
      c.arc(sx, sy, lockSize * 0.85, t * 2, t * 2 + Math.PI * 2);
      c.stroke();

      c.strokeStyle = 'rgba(52, 211, 153, 0.8)';
      c.setLineDash([3, 4]);
      c.beginPath();
      c.arc(sx, sy, lockSize * 1.25, -t * 2.5, -t * 2.5 + Math.PI * 2);
      c.stroke();

      // White Corner Brackets
      c.setLineDash([]);
      c.strokeStyle = '#ffffff';
      c.lineWidth = 2.0;
      const b = lockSize * 1.35;
      const len = 7;

      c.beginPath();
      c.moveTo(sx - b + len, sy - b);
      c.lineTo(sx - b, sy - b);
      c.lineTo(sx - b, sy - b + len);
      c.stroke();

      c.beginPath();
      c.moveTo(sx + b - len, sy - b);
      c.lineTo(sx + b, sy - b);
      c.lineTo(sx + b, sy - b + len);
      c.stroke();

      c.beginPath();
      c.moveTo(sx - b + len, sy + b);
      c.lineTo(sx - b, sy + b);
      c.lineTo(sx - b, sy + b - len);
      c.stroke();

      c.beginPath();
      c.moveTo(sx + b - len, sy + b);
      c.lineTo(sx + b, sy + b);
      c.lineTo(sx + b, sy + b - len);
      c.stroke();

      // Shockwave ring in emerald
      const ripple = (t * 2) % 1;
      c.strokeStyle = `rgba(16, 185, 129, ${1 - ripple})`;
      c.lineWidth = 1.2;
      c.beginPath();
      c.arc(sx, sy, lockSize * 0.8 + ripple * 28, 0, Math.PI * 2);
      c.stroke();

      // Leader line & Callout Card in Emerald & White
      const cardX = sx + 45 > cx + 60 ? sx - 170 : sx + 40;
      const cardY = sy - 36;

      c.strokeStyle = 'rgba(16, 185, 129, 0.65)';
      c.lineWidth = 1.0;
      c.setLineDash([2, 2]);
      c.beginPath();
      c.moveTo(sx + (cardX > sx ? lockSize * 1.2 : -lockSize * 1.2), sy);
      c.lineTo(cardX > sx ? cardX : cardX + 155, cardY + 18);
      c.stroke();

      c.setLineDash([]);
      c.fillStyle = 'rgba(2, 24, 16, 0.92)';
      c.strokeStyle = 'rgba(16, 185, 129, 0.75)';
      c.lineWidth = 1.0;
      c.beginPath();
      c.roundRect(cardX, cardY, 155, 48, 6);
      c.fill();
      c.stroke();

      c.fillStyle = '#34d399';
      c.font = 'bold 9px monospace';
      c.fillText(`⚡ OPTIMAL ACTION CANDIDATE`, cardX + 8, cardY + 13);

      c.fillStyle = '#ffffff';
      c.font = 'bold 11px monospace';
      c.fillText(`${atom.id}`, cardX + 8, cardY + 26);

      c.fillStyle = '#a7f3d0';
      c.font = '8px monospace';
      c.fillText(`VoI: +${atom.score.toFixed(3)} | LOCKED`, cardX + 8, cardY + 39);

      c.restore();
    };

    const drawHoverHUD = (c: CanvasRenderingContext2D, atom: AtomNode) => {
      c.save();
      const hx = atom.sx + 15;
      const hy = atom.sy - 25;

      c.fillStyle = 'rgba(2, 24, 16, 0.95)';
      c.strokeStyle = 'rgba(52, 211, 153, 0.6)';
      c.lineWidth = 1;
      c.beginPath();
      c.roundRect(hx, hy, 140, 42, 4);
      c.fill();
      c.stroke();

      c.fillStyle = '#34d399';
      c.font = 'bold 9px monospace';
      c.fillText(`// CANDIDATE NODE`, hx + 6, hy + 13);

      c.fillStyle = '#ffffff';
      c.font = 'bold 10px monospace';
      c.fillText(atom.id, hx + 6, hy + 25);

      c.fillStyle = '#a7f3d0';
      c.font = '8px monospace';
      c.fillText(atom.composition.slice(0, 22), hx + 6, hy + 36);
      c.restore();
    };

    const render = () => {
      time += 0.02;
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const sphereRadius = Math.min(width, height) * 0.38;
      const fov = 420;

      if (autoRotate && !isDraggingRef.current) {
        rotationRef.current.x += velocityRef.current.x;
        rotationRef.current.y += velocityRef.current.y;
      }

      const rx = rotationRef.current.x;
      const ry = rotationRef.current.y;
      const rz = rotationRef.current.z;

      const cosX = Math.cos(rx), sinX = Math.sin(rx);
      const cosY = Math.cos(ry), sinY = Math.sin(ry);
      const cosZ = Math.cos(rz), sinZ = Math.sin(rz);

      // 1. Deep Obsidian Forest Background Glow
      const bgGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, sphereRadius * 1.5);
      bgGrad.addColorStop(0, 'rgba(3, 43, 30, 0.55)');
      bgGrad.addColorStop(0.5, 'rgba(2, 24, 16, 0.3)');
      bgGrad.addColorStop(1, 'rgba(1, 13, 9, 0)');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      // 2. Central Emerald Core
      const corePulse = 1 + Math.sin(time * 3) * 0.12;
      const coreGrad = ctx.createRadialGradient(cx, cy, 2, cx, cy, 38 * corePulse);
      coreGrad.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
      coreGrad.addColorStop(0.2, 'rgba(16, 185, 129, 0.85)');
      coreGrad.addColorStop(0.6, 'rgba(5, 150, 105, 0.25)');
      coreGrad.addColorStop(1, 'rgba(1, 13, 9, 0)');
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, 38 * corePulse, 0, Math.PI * 2);
      ctx.fill();

      // Pulsing inner ring
      ctx.save();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 6]);
      ctx.beginPath();
      ctx.arc(cx, cy, 24 * corePulse, time, time + Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      const atoms = atomsRef.current;
      const laserScanY = Math.sin(time * 1.8) * sphereRadius;

      for (let i = 0; i < atoms.length; i++) {
        const a = atoms[i];

        const x1 = a.x0 * cosY + a.z0 * sinY;
        const y1 = a.y0;
        const z1 = -a.x0 * sinY + a.z0 * cosY;

        const x2 = x1;
        const y2 = y1 * cosX - z1 * sinX;
        const z2 = y1 * sinX + z1 * cosX;

        const x3 = x2 * cosZ - y2 * sinZ;
        const y3 = x2 * sinZ + y2 * cosZ;
        const z3 = z2;

        a.x = x3 * sphereRadius;
        a.y = y3 * sphereRadius;
        a.z = z3 * sphereRadius;

        const scale = fov / (fov + a.z);
        a.scale = scale;
        a.sx = cx + a.x * scale;
        a.sy = cy + a.y * scale;

        a.electronAngle += 0.05 + (i % 4) * 0.01;
      }

      const sortedAtoms = [...atoms].sort((a, b) => a.z - b.z);

      drawHolographicRings(ctx, cx, cy, sphereRadius, rx, ry, true);

      // Crystalline Lattice Connections (Front)
      ctx.save();
      ctx.strokeStyle = 'rgba(16, 185, 129, 0.12)';
      ctx.lineWidth = 0.8;
      for (let i = 0; i < sortedAtoms.length; i += 3) {
        const a1 = sortedAtoms[i];
        if (a1.z < 0) continue;
        for (let j = i + 1; j < Math.min(i + 4, sortedAtoms.length); j++) {
          const a2 = sortedAtoms[j];
          if (a2.z < 0) continue;
          const dx = a1.sx - a2.sx;
          const dy = a1.sy - a2.sy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 48) {
            ctx.beginPath();
            ctx.moveTo(a1.sx, a1.sy);
            ctx.lineTo(a2.sx, a2.sy);
            ctx.stroke();
          }
        }
      }
      ctx.restore();

      let winnerAtom: AtomNode | null = null;

      for (let i = 0; i < sortedAtoms.length; i++) {
        const a = sortedAtoms[i];
        const isBack = a.z < 0;
        const depthAlpha = Math.max(0.12, Math.min(1.0, (a.z + sphereRadius) / (2 * sphereRadius)));
        const r = Math.max(1.8, a.radius * a.scale);

        const nearScan = Math.abs((a.sy - cy) - laserScanY) < 14;
        const flashIntensity = nearScan ? 1.8 : 1.0;

        if (a.isWinner) {
          winnerAtom = a;
        }

        ctx.save();
        ctx.globalAlpha = isBack ? depthAlpha * 0.45 : depthAlpha;

        const glowR = r * (a.isWinner ? 4.5 : a.isPareto ? 3.0 : 2.0) * flashIntensity;
        const grad = ctx.createRadialGradient(a.sx, a.sy, r * 0.2, a.sx, a.sy, glowR);
        grad.addColorStop(0, a.color);
        grad.addColorStop(0.4, a.glowColor);
        grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(a.sx, a.sy, glowR, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = a.isWinner ? '#ffffff' : nearScan ? '#ffffff' : a.color;
        ctx.beginPath();
        ctx.arc(a.sx, a.sy, r, 0, Math.PI * 2);
        ctx.fill();

        // Mini Electron Orbit
        if (!isBack && (a.isPareto || a.isWinner)) {
          ctx.strokeStyle = a.isWinner ? 'rgba(255, 255, 255, 0.7)' : 'rgba(16, 185, 129, 0.35)';
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          const orbitR = r * 2.4;
          ctx.ellipse(a.sx, a.sy, orbitR, orbitR * 0.45, a.electronAngle, 0, Math.PI * 2);
          ctx.stroke();

          const ex = a.sx + Math.cos(a.electronAngle * 2) * orbitR;
          const ey = a.sy + Math.sin(a.electronAngle * 2) * orbitR * 0.45;
          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(ex, ey, 1.2, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.restore();
      }

      drawHolographicRings(ctx, cx, cy, sphereRadius, rx, ry, false);

      // Emerald Laser Scanning Plane
      ctx.save();
      const beamY = cy + laserScanY;
      const beamWidth = sphereRadius * 2.2;
      const beamGrad = ctx.createLinearGradient(cx - beamWidth / 2, beamY, cx + beamWidth / 2, beamY);
      beamGrad.addColorStop(0, 'rgba(16, 185, 129, 0)');
      beamGrad.addColorStop(0.2, 'rgba(16, 185, 129, 0.45)');
      beamGrad.addColorStop(0.5, 'rgba(255, 255, 255, 0.95)');
      beamGrad.addColorStop(0.8, 'rgba(16, 185, 129, 0.45)');
      beamGrad.addColorStop(1, 'rgba(16, 185, 129, 0)');

      ctx.fillStyle = beamGrad;
      ctx.fillRect(cx - beamWidth / 2, beamY - 1, beamWidth, 2);

      ctx.fillStyle = 'rgba(5, 150, 105, 0.08)';
      ctx.fillRect(cx - beamWidth / 2, beamY - 8, beamWidth, 16);
      ctx.restore();

      if (winnerAtom && winnerAtom.z > -sphereRadius * 0.3) {
        drawStarkTargetLock(ctx, winnerAtom, cx, cy, time);
      }

      if (hoveredAtom) {
        drawHoverHUD(ctx, hoveredAtom);
      }

      animFrameIdRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animFrameIdRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, [autoRotate, winnerId, activeTargetId, hoveredAtom]);

  const handlePointerDown = (e: React.PointerEvent) => {
    isDraggingRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (isDraggingRef.current) {
      const dx = e.clientX - lastMouseRef.current.x;
      const dy = e.clientY - lastMouseRef.current.y;

      rotationRef.current.y += dx * 0.008;
      rotationRef.current.x += dy * 0.008;

      velocityRef.current = { x: dy * 0.002, y: dx * 0.002 };
      lastMouseRef.current = { x: e.clientX, y: e.clientY };
    } else {
      const hovered = atomsRef.current.find((a: AtomNode) => {
        if (a.z < 0) return false;
        const dist = Math.hypot(a.sx - mx, a.sy - my);
        return dist < 12;
      });
      setHoveredAtom(hovered || null);
    }
  };

  const handlePointerUp = () => {
    isDraggingRef.current = false;
  };

  const handleClick = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const clicked = atomsRef.current.find((a: AtomNode) => {
      if (a.z < 0) return false;
      const dist = Math.hypot(a.sx - mx, a.sy - my);
      return dist < 14;
    });

    if (clicked) {
      setActiveTargetId(clicked.id);
      if (onSelectCandidate) {
        onSelectCandidate(clicked.id);
      }
    }
  };

  return (
    <div 
      ref={containerRef}
      className={`relative rounded-xl border border-emerald-900/40 bg-slate-950 overflow-hidden shadow-xl transition-all ${
        isFullscreen ? 'fixed inset-4 z-50 flex flex-col' : 'w-full h-[450px]'
      }`}
      style={{
        backgroundImage: `radial-gradient(circle at 50% 50%, #032b1e 0%, #01120c 90%)`,
      }}
    >
      {/* Top Bar in Clean Emerald & White */}
      <div className="absolute top-0 left-0 right-0 z-10 px-4 py-2.5 flex items-center justify-between border-b border-emerald-900/40 bg-slate-950/70 backdrop-blur-xs">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold tracking-widest text-emerald-400">
                STARK ATOMIC LATTICE
              </span>
              <span className="text-2xs font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/60">
                EMERALD SPHERE
              </span>
            </div>
            <span className="text-3xs font-mono text-slate-400 tracking-wider">
              {scanStatusText}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={triggerStarkScanSequence}
            disabled={scanSequenceActive}
            className={`px-3 py-1.5 rounded-lg font-mono text-xs font-bold flex items-center gap-1.5 transition shadow-sm cursor-pointer ${
              scanSequenceActive 
                ? 'bg-emerald-900/50 text-emerald-300 border border-emerald-600 animate-pulse'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white font-semibold'
            }`}
          >
            <Zap className={`w-3.5 h-3.5 ${scanSequenceActive ? 'animate-spin' : ''}`} />
            <span>{scanSequenceActive ? 'SCANNING...' : '⚡ SYNTHESIZE NEXT ELEMENT'}</span>
          </button>

          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`p-1.5 rounded-lg border transition cursor-pointer ${
              autoRotate 
                ? 'bg-emerald-950/80 border-emerald-700 text-emerald-300' 
                : 'bg-slate-900 border-slate-800 text-slate-500'
            }`}
            title={autoRotate ? 'Pause Orbit' : 'Resume Auto Orbit'}
          >
            <RotateCw className={`w-3.5 h-3.5 ${autoRotate ? 'animate-spin' : ''}`} style={{ animationDuration: '8s' }} />
          </button>

          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 rounded-lg border border-emerald-900/60 bg-emerald-950/60 text-emerald-400 hover:text-white transition cursor-pointer"
            title={isFullscreen ? 'Exit Fullscreen' : 'Expand Theater Mode'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onClick={handleClick}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      />

      {/* Bottom Legend Bar in Emerald & White */}
      <div className="absolute bottom-0 left-0 right-0 z-10 px-4 py-2 border-t border-emerald-900/30 bg-slate-950/80 backdrop-blur-xs flex flex-wrap items-center justify-between text-2xs font-mono text-slate-400">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-white ring-2 ring-emerald-400 shadow-sm shadow-emerald-400" />
            <span className="text-white font-bold">Optimal Target ({activeTargetId})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-slate-300">High VoI Candidate</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-700" />
            <span className="text-slate-300">Tested (Synthesized)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-200/80" />
            <span className="text-slate-300">Candidate Lattice</span>
          </div>
        </div>

        <div className="hidden sm:flex items-center gap-3 text-3xs text-emerald-400/80">
          <span>DRAG TO ORBIT</span>
          <span>•</span>
          <span>CLICK ATOM TO TARGET</span>
        </div>
      </div>
    </div>
  );
};
