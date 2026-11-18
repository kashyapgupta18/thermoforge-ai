import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface HeatFluxVector {
  origin: number[];
  direction: number[];
  magnitude: number;
}

interface Viewport3DProps {
  surfaceMesh: {
    positions: number[];
    normals: number[];
    temperatures: number[];
    heat_fluxes?: number[];
    wireframe_lines: number[];
    num_triangles?: number;
    heat_source_centroids?: number[][];
    heat_flux_vectors?: HeatFluxVector[];
    heat_flux_range?: {
      q_min_w_m2: number;
      q_max_w_m2: number;
      q_avg_w_m2?: number;
    };
    hotspot?: {
      coordinate: number[];
      peak_temperature: number;
    };
  } | null;
  vizMode: 'solid' | 'wireframe' | 'temperature' | 'heat_flux' | 'hotspot' | 'cross_section';
  tMin: number;
  tMax: number;
  showFluxVectors?: boolean;
  onProbeTemp?: (temp: number | null, coord: string) => void;
}

export const Viewport3D: React.FC<Viewport3DProps> = ({
  surfaceMesh,
  vizMode,
  tMin,
  tMax,
  showFluxVectors = false,
  onProbeTemp,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const vectorsGroupRef = useRef<THREE.Group | null>(null);
  const probeMarkerRef = useRef<THREE.Mesh | null>(null);
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());
  const surfaceMeshRef = useRef(surfaceMesh);
  surfaceMeshRef.current = surfaceMesh;

  const [probedData, setProbedData] = useState<{
    temp: number;
    flux: number | null;
    x: number;
    y: number;
    z: number;
  } | null>(null);
  const [cameraPreset, setCameraPreset] = useState<string>('iso');
  const [tempFilterPct, setTempFilterPct] = useState<number>(0);

  // Colormap for Temperature (°C): Blue -> Cyan -> Green -> Yellow -> Red
  const getTemperatureColor = (val: number, min: number, max: number): THREE.Color => {
    const t = Math.max(0, Math.min(1, (val - min) / (max - min || 1e-6)));
    let r = 0, g = 0, b = 0;
    if (t < 0.25) {
      r = 0;
      g = 4 * t;
      b = 1;
    } else if (t < 0.5) {
      r = 0;
      g = 1;
      b = 1 - 4 * (t - 0.25);
    } else if (t < 0.75) {
      r = 4 * (t - 0.5);
      g = 1;
      b = 0;
    } else {
      r = 1;
      g = 1 - 4 * (t - 0.75);
      b = 0;
    }
    return new THREE.Color(r, g, b);
  };

  // Colormap for Heat Flux Magnitude |q| = |-k ∇T| (W/m²): Deep Indigo -> Teal -> Cyan -> Amber/White
  const getHeatFluxColor = (val: number, min: number, max: number): THREE.Color => {
    const t = Math.max(0, Math.min(1, (val - min) / (max - min || 1e-6)));
    if (t < 0.35) {
      const s = t / 0.35;
      return new THREE.Color(0.06 * (1 - s) + 0.05 * s, 0.09 * (1 - s) + 0.55 * s, 0.20 * (1 - s) + 0.65 * s);
    } else if (t < 0.75) {
      const s = (t - 0.35) / 0.40;
      return new THREE.Color(0.05 * (1 - s) + 0.15 * s, 0.55 * (1 - s) + 0.85 * s, 0.65 * (1 - s) + 0.95 * s);
    } else {
      const s = (t - 0.75) / 0.25;
      return new THREE.Color(0.15 * (1 - s) + 1.0 * s, 0.85 * (1 - s) + 0.95 * s, 0.95 * (1 - s) + 0.55 * s);
    }
  };

  useEffect(() => {
    if (!containerRef.current) return;
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e131f);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(7, 6, 8);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.localClippingEnabled = true;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight1.position.set(12, 18, 12);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
    dirLight2.position.set(-10, -5, -10);
    scene.add(dirLight2);

    const grid = new THREE.GridHelper(10, 20, 0x1e293b, 0x1e293b);
    grid.position.y = -2.2;
    scene.add(grid);

    const meshGroup = new THREE.Group();
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    const vectorsGroup = new THREE.Group();
    scene.add(vectorsGroup);
    vectorsGroupRef.current = vectorsGroup;

    const markerGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0xff0055, wireframe: true });
    const probeMarker = new THREE.Mesh(markerGeo, markerMat);
    probeMarker.visible = false;
    scene.add(probeMarker);
    probeMarkerRef.current = probeMarker;

    let isDragging = false;
    let downPos = { x: 0, y: 0 };
    let prevMouse = { x: 0, y: 0 };
    const spherical = { radius: 12, theta: Math.PI / 4, phi: Math.PI / 3 };

    const updateCameraPos = () => {
      camera.position.x = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
      camera.position.y = spherical.radius * Math.cos(spherical.phi);
      camera.position.z = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
      camera.lookAt(0, 0, 0);
    };
    updateCameraPos();

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      downPos = { x: e.clientX, y: e.clientY };
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        const dx = e.clientX - prevMouse.x;
        const dy = e.clientY - prevMouse.y;
        spherical.theta -= dx * 0.008;
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi - dy * 0.008));
        updateCameraPos();
        prevMouse = { x: e.clientX, y: e.clientY };
      }
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onWheel = (e: WheelEvent) => {
      spherical.radius = Math.max(3, Math.min(25, spherical.radius + e.deltaY * 0.01));
      updateCameraPos();
    };

    const onClick = (e: MouseEvent) => {
      const moveDist = Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y);
      if (!containerRef.current || !meshGroupRef.current || moveDist > 5) return;
      const rect = containerRef.current.getBoundingClientRect();
      mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);
      const intersects = raycasterRef.current.intersectObjects(meshGroupRef.current.children, true);

      if (intersects.length > 0) {
        const hit = intersects[0];
        const pt = hit.point;
        const sm = surfaceMeshRef.current;
        if (hit.face && sm && sm.temperatures) {
          const vIdx = hit.face.a;
          const exactTemp = sm.temperatures[vIdx] ?? tMin;
          const exactFlux = sm.heat_fluxes && sm.heat_fluxes[vIdx] !== undefined ? sm.heat_fluxes[vIdx] : null;
          setProbedData({
            temp: exactTemp,
            flux: exactFlux,
            x: Number(pt.x.toFixed(2)),
            y: Number(pt.y.toFixed(2)),
            z: Number(pt.z.toFixed(2))
          });
          if (probeMarkerRef.current) {
            probeMarkerRef.current.position.copy(pt);
            probeMarkerRef.current.visible = true;
          }
          if (onProbeTemp) onProbeTemp(exactTemp, `(${pt.x.toFixed(2)}, ${pt.y.toFixed(2)}, ${pt.z.toFixed(2)})`);
        }
      }
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    dom.addEventListener('wheel', onWheel);
    dom.addEventListener('click', onClick);

    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      dom.removeEventListener('wheel', onWheel);
      dom.removeEventListener('click', onClick);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  const applyCameraPreset = (preset: 'iso' | 'top' | 'front' | 'side') => {
    if (!cameraRef.current) return;
    setCameraPreset(preset);
    const camera = cameraRef.current;
    if (preset === 'top') {
      camera.position.set(0, 14, 0.01);
    } else if (preset === 'front') {
      camera.position.set(0, 0, 14);
    } else if (preset === 'side') {
      camera.position.set(14, 0, 0);
    } else {
      camera.position.set(7, 6, 8);
    }
    camera.lookAt(0, 0, 0);
  };

  const exportImage = () => {
    if (!rendererRef.current) return;
    const dataURL = rendererRef.current.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataURL;
    a.download = `thermoforge_simulation_${vizMode}_${Date.now()}.png`;
    a.click();
  };

  const qMin = surfaceMesh?.heat_flux_range?.q_min_w_m2 ?? 0;
  const qMax = surfaceMesh?.heat_flux_range?.q_max_w_m2 ?? 1;
  const qAvg = surfaceMesh?.heat_flux_range?.q_avg_w_m2 ?? (qMin + qMax) * 0.5;
  const filterThresholdTemp = tMin + (tMax - tMin) * (tempFilterPct / 100);

  // Re-build 3D Mesh when surfaceMesh, vizMode, or filter changes
  useEffect(() => {
    if (!meshGroupRef.current || !surfaceMesh) return;
    const group = meshGroupRef.current;
    while (group.children.length > 0) {
      group.remove(group.children[0]);
    }

    const { positions, normals, temperatures, heat_fluxes, wireframe_lines, hotspot } = surfaceMesh;

    const triGeo = new THREE.BufferGeometry();
    const posArr = new Float32Array(positions);
    const normArr = new Float32Array(normals);
    triGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    triGeo.setAttribute('normal', new THREE.BufferAttribute(normArr, 3));

    const numVerts = positions.length / 3;
    const colArr = new Float32Array(numVerts * 3);

    for (let i = 0; i < numVerts; i++) {
      const temp = temperatures[i] ?? tMin;
      const flux = heat_fluxes && heat_fluxes[i] !== undefined ? heat_fluxes[i] : qMin;
      let col: THREE.Color;
      if (vizMode === 'solid') {
        col = new THREE.Color(0x94a3b8);
      } else if (vizMode === 'hotspot') {
        const isHot = (temp - tMin) / (tMax - tMin || 1) >= 0.8;
        col = isHot ? new THREE.Color(0xff1e56) : new THREE.Color(0x1e293b);
      } else if (vizMode === 'heat_flux') {
        // Actual computed heat flux magnitude |q| = |-k ∇T| in W/m²
        col = getHeatFluxColor(flux, qMin, qMax);
      } else {
        if (tempFilterPct > 0 && temp < filterThresholdTemp) {
          col = new THREE.Color(0x1e293b);
        } else {
          col = getTemperatureColor(temp, tMin, tMax);
        }
      }
      colArr[i * 3] = col.r;
      colArr[i * 3 + 1] = col.g;
      colArr[i * 3 + 2] = col.b;
    }
    triGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));

    const triMat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.35,
      metalness: 0.25,
      side: THREE.DoubleSide,
      transparent: vizMode === 'wireframe',
      opacity: vizMode === 'wireframe' ? 0.15 : 1.0,
      clippingPlanes: vizMode === 'cross_section' ? [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.1)] : []
    });

    const triMesh = new THREE.Mesh(triGeo, triMat);
    group.add(triMesh);

    if (vizMode === 'wireframe' || vizMode === 'solid' || vizMode === 'temperature' || vizMode === 'heat_flux') {
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(wireframe_lines), 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: vizMode === 'wireframe' ? 0x38bdf8 : 0x0f172a,
        transparent: true,
        opacity: vizMode === 'wireframe' ? 0.9 : 0.22
      });
      const lines = new THREE.LineSegments(lineGeo, lineMat);
      group.add(lines);
    }

    if (vizMode === 'hotspot' && hotspot && probeMarkerRef.current) {
      probeMarkerRef.current.position.set(hotspot.coordinate[0], hotspot.coordinate[1], hotspot.coordinate[2]);
      probeMarkerRef.current.visible = true;
    }

    if (vectorsGroupRef.current) {
      const vGroup = vectorsGroupRef.current;
      while (vGroup.children.length > 0) {
        vGroup.remove(vGroup.children[0]);
      }

      if ((vizMode === 'heat_flux' || showFluxVectors) && surfaceMesh.heat_flux_vectors) {
        surfaceMesh.heat_flux_vectors.forEach((vec) => {
          const origin = new THREE.Vector3(vec.origin[0], vec.origin[1], vec.origin[2]);
          const dir = new THREE.Vector3(vec.direction[0], vec.direction[1], vec.direction[2]);
          const normMag = Math.max(0, Math.min(1, (vec.magnitude - qMin) / (qMax - qMin || 1)));
          const len = 0.22 + 0.28 * normMag;
          const arrowCol = getHeatFluxColor(vec.magnitude, qMin, qMax).getHex();
          const arrow = new THREE.ArrowHelper(dir, origin, len, arrowCol, 0.11, 0.065);
          vGroup.add(arrow);
        });
      }
    }
  }, [surfaceMesh, vizMode, showFluxVectors, tMin, tMax, qMin, qMax, tempFilterPct, filterThresholdTemp]);

  return (
    <div className="relative w-full h-full select-none overflow-hidden rounded-xl border border-graphite-800 bg-graphite-950">
      <div ref={containerRef} className="w-full h-full cursor-crosshair active:cursor-grabbing" />

      {/* Mode-Specific Calibrated Engineering Legend */}
      <div className="absolute top-4 right-4 bg-graphite-900/95 backdrop-blur-md p-3.5 rounded-lg border border-graphite-700 shadow-xl flex flex-col items-center min-w-[148px]">
        {vizMode === 'heat_flux' ? (
          <>
            <span className="text-[10px] font-bold tracking-wider text-cyan-300 mb-1">HEAT FLUX |q| (W/m²)</span>
            <div className="flex items-center space-x-2">
              <div className="flex flex-col justify-between h-36 text-[10px] font-mono text-right">
                <span className="font-bold text-yellow-200">{qMax.toFixed(0)} W/m²</span>
                <span className="text-cyan-300">{qAvg.toFixed(0)} W/m²</span>
                <span className="text-slate-400">{qMin.toFixed(0)} W/m²</span>
              </div>
              <div className="w-4 h-36 rounded-sm bg-gradient-to-t from-slate-900 via-teal-600 via-cyan-400 to-yellow-200 border border-graphite-700 shadow-inner" />
            </div>
            <div className="mt-2 text-[9px] text-cyan-400/90 uppercase tracking-widest font-mono">q = -k·∇T (Fourier)</div>
          </>
        ) : vizMode === 'hotspot' ? (
          <>
            <span className="text-[10px] font-bold tracking-wider text-rose-400 mb-1">HOTSPOT ISOLATION (°C)</span>
            <div className="flex items-center space-x-2">
              <div className="flex flex-col justify-between h-32 text-[10px] font-mono text-right">
                <span className="font-bold text-rose-400">Peak: {tMax.toFixed(2)}°C</span>
                <span className="text-rose-300"> Cutoff: {(tMin + (tMax - tMin) * 0.8).toFixed(2)}°C</span>
                <span className="text-gray-500">Base: {tMin.toFixed(2)}°C</span>
              </div>
              <div className="w-4 h-32 rounded-sm bg-gradient-to-t from-slate-800 via-slate-800 to-rose-600 border border-graphite-700 shadow-inner" />
            </div>
            <div className="mt-2 text-[9px] text-rose-400/80 uppercase tracking-widest font-mono">Top 20% Thermal Rise</div>
          </>
        ) : vizMode === 'solid' ? (
          <>
            <span className="text-[10px] font-bold tracking-wider text-gray-300 mb-1">SOLID CAD GEOMETRY</span>
            <div className="w-28 py-2 px-2 rounded bg-slate-400/20 border border-slate-500 text-center text-[10px] text-gray-200 font-mono">
              3D STEPPED FINNED ASSEMBLY
            </div>
            <div className="mt-2 text-[9px] text-gray-400 font-mono">
              Triangles: {surfaceMesh?.num_triangles ?? 0}
            </div>
          </>
        ) : vizMode === 'wireframe' ? (
          <>
            <span className="text-[10px] font-bold tracking-wider text-cyan-400 mb-1">FEM HEX-8 MESH</span>
            <div className="w-28 py-2 px-2 rounded bg-cyan-950/40 border border-cyan-500/50 text-center text-[10px] text-cyan-300 font-mono">
              TRILINEAR HEXAHEDRA
            </div>
            <div className="mt-2 text-[9px] text-gray-400 font-mono">
              Boundary Faces: {(surfaceMesh?.num_triangles ?? 0) / 2}
            </div>
          </>
        ) : (
          <>
            <span className="text-[10px] font-bold tracking-wider text-gray-300 mb-1">
              {vizMode === 'cross_section' ? 'SECTION CONTOUR (°C)' : 'TEMPERATURE (°C)'}
            </span>
            <div className="flex items-center space-x-2">
              <div className="flex flex-col justify-between h-36 text-[10px] font-mono text-right">
                <span className="font-bold text-rose-400">{tMax.toFixed(2)}°C</span>
                <span className="text-emerald-300">{((tMin + tMax) * 0.5).toFixed(2)}°C</span>
                <span className="font-bold text-blue-400">{tMin.toFixed(2)}°C</span>
              </div>
              <div className="w-4 h-36 rounded-sm bg-gradient-to-t from-blue-600 via-cyan-400 via-yellow-400 to-red-600 border border-graphite-700 shadow-inner" />
            </div>
            <div className="w-full mt-2 pt-2 border-t border-graphite-800">
              <div className="flex justify-between text-[9px] font-mono text-gray-400">
                <span>Filter &ge;</span>
                <span className="text-cyan-300">{filterThresholdTemp.toFixed(1)}°C</span>
              </div>
              <input
                type="range"
                min="0"
                max="90"
                step="5"
                value={tempFilterPct}
                onChange={(e) => setTempFilterPct(Number(e.target.value))}
                className="w-full h-1 accent-cyan-500 cursor-pointer"
              />
            </div>
          </>
        )}
      </div>

      {/* Probed Node Info Box (Exact FEM Nodal Lookup) */}
      {probedData && (
        <div className="absolute top-4 left-4 bg-graphite-900/95 backdrop-blur-md p-3 rounded-lg border border-cyan-500/50 shadow-xl text-xs font-mono text-gray-200">
          <div className="text-cyan-400 font-bold mb-1 flex items-center justify-between">
            <span>FEM SURFACE PROBE</span>
            <button
              onClick={() => setProbedData(null)}
              className="text-gray-400 hover:text-white ml-3 text-[10px] cursor-pointer"
            >
              ✕
            </button>
          </div>
          <div>Temperature: <strong className="text-rose-400">{probedData.temp.toFixed(3)} °C</strong></div>
          {probedData.flux !== null && (
            <div>Heat Flux |q|: <strong className="text-cyan-300">{probedData.flux.toFixed(1)} W/m²</strong></div>
          )}
          <div className="text-[10px] text-gray-400 mt-0.5">Coord: ({probedData.x}, {probedData.y}, {probedData.z})</div>
        </div>
      )}

      {/* Camera Presets & Screenshot Toolbar */}
      <div className="absolute bottom-4 right-4 bg-graphite-900/90 backdrop-blur-md p-1.5 rounded-lg border border-graphite-700 flex items-center space-x-1 text-xs">
        <span className="text-[10px] text-gray-400 px-2 font-mono">Camera:</span>
        {(['iso', 'top', 'front', 'side'] as const).map((cam) => (
          <button
            key={cam}
            onClick={() => applyCameraPreset(cam)}
            className={`px-2 py-1 rounded text-[10px] font-mono uppercase transition-all ${
              cameraPreset === cam ? 'bg-cyan-600 text-white font-bold' : 'text-gray-400 hover:text-white'
            }`}
          >
            {cam}
          </button>
        ))}
        <span className="text-gray-600">|</span>
        <button
          onClick={exportImage}
          className="px-2.5 py-1 bg-graphite-800 hover:bg-graphite-700 text-cyan-300 rounded text-[10px] font-mono transition-all"
        >
          📷 Export PNG
        </button>
      </div>

      {/* Floating HUD */}
      <div className="absolute bottom-4 left-4 bg-graphite-900/80 backdrop-blur-md px-3 py-1.5 rounded-md border border-graphite-700 text-xs text-gray-300 flex items-center space-x-3 pointer-events-none">
        <span className="flex items-center space-x-1">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="font-mono text-cyan-300">Click Surface: Probe Exact T &amp; |q|</span>
        </span>
        <span className="text-gray-600">|</span>
        <span className="font-mono text-gray-400">Drag: Orbit • Wheel: Zoom</span>
      </div>
    </div>
  );
};
