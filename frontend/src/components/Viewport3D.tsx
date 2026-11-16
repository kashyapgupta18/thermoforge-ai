import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface Viewport3DProps {
  surfaceMesh: {
    positions: number[];
    normals: number[];
    temperatures: number[];
    wireframe_lines: number[];
    heat_source_centroids?: number[][];
  } | null;
  vizMode: 'solid' | 'wireframe' | 'temperature' | 'hotspot' | 'cross_section';
  tMin: number;
  tMax: number;
  onProbeTemp?: (temp: number | null, coord: string) => void;
}

export const Viewport3D: React.FC<Viewport3DProps> = ({
  surfaceMesh,
  vizMode,
  tMin,
  tMax,
  onProbeTemp,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const meshGroupRef = useRef<THREE.Group | null>(null);
  const probeMarkerRef = useRef<THREE.Mesh | null>(null);

  const [hoveredTemp, setHoveredTemp] = useState<number | null>(null);

  // Helper function to map temperature to jet/turbo colormap (Blue -> Cyan -> Green -> Yellow -> Red)
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

  useEffect(() => {
    if (!containerRef.current) return;
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    // 1. Scene Setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0e131f);
    sceneRef.current = scene;

    // 2. Camera Setup
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(7, 6, 8);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // 3. Renderer Setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(10, 15, 10);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.4);
    dirLight2.position.set(-10, -5, -10);
    scene.add(dirLight2);

    // 5. Grid Ground Plane
    const grid = new THREE.GridHelper(10, 20, 0x1e293b, 0x1e293b);
    grid.position.y = -2.0;
    scene.add(grid);

    // Group for geometry
    const meshGroup = new THREE.Group();
    scene.add(meshGroup);
    meshGroupRef.current = meshGroup;

    // Hotspot / Probe marker sphere
    const markerGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const markerMat = new THREE.MeshBasicMaterial({ color: 0xff0055, wireframe: true });
    const probeMarker = new THREE.Mesh(markerGeo, markerMat);
    probeMarker.visible = false;
    scene.add(probeMarker);
    probeMarkerRef.current = probeMarker;

    // Mouse Interaction: Orbit Controls
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let spherical = { radius: 12, theta: Math.PI / 4, phi: Math.PI / 3 };

    const updateCameraPos = () => {
      camera.position.x = spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta);
      camera.position.y = spherical.radius * Math.cos(spherical.phi);
      camera.position.z = spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta);
      camera.lookAt(0, 0, 0);
    };
    updateCameraPos();

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
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

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    dom.addEventListener('wheel', onWheel);

    // Animation Loop
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
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update Mesh Buffers when surfaceMesh or vizMode changes
  useEffect(() => {
    if (!meshGroupRef.current || !surfaceMesh) return;
    const group = meshGroupRef.current;
    while (group.children.length > 0) {
      group.remove(group.children[0]);
    }

    const { positions, normals, temperatures, wireframe_lines, heat_source_centroids } = surfaceMesh;

    // 1. Triangular Surface Mesh
    const triGeo = new THREE.BufferGeometry();
    const posArr = new Float32Array(positions);
    const normArr = new Float32Array(normals);
    triGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    triGeo.setAttribute('normal', new THREE.BufferAttribute(normArr, 3));

    // Per-vertex colors
    const numVerts = positions.length / 3;
    const colArr = new Float32Array(numVerts * 3);

    for (let i = 0; i < numVerts; i++) {
      const temp = temperatures[i] || tMin;
      let col: THREE.Color;
      if (vizMode === 'solid') {
        col = new THREE.Color(0x94a3b8); // Metallic slate
      } else if (vizMode === 'hotspot') {
        // Highlight temperatures > 85% range
        const isHot = (temp - tMin) / (tMax - tMin || 1) > 0.75;
        col = isHot ? new THREE.Color(0xff1e56) : new THREE.Color(0x334155);
      } else {
        col = getTemperatureColor(temp, tMin, tMax);
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
      clippingPlanes: vizMode === 'cross_section' ? [new THREE.Plane(new THREE.Vector3(1, 0, 0), 0.2)] : []
    });

    const triMesh = new THREE.Mesh(triGeo, triMat);
    group.add(triMesh);

    // 2. Mesh Wireframe Overlay
    if (vizMode === 'wireframe' || vizMode === 'solid' || vizMode === 'temperature') {
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(wireframe_lines), 3));
      const lineMat = new THREE.LineBasicMaterial({
        color: vizMode === 'wireframe' ? 0x38bdf8 : 0x0f172a,
        transparent: true,
        opacity: vizMode === 'wireframe' ? 0.9 : 0.25
      });
      const lines = new THREE.LineSegments(lineGeo, lineMat);
      group.add(lines);
    }

    // 3. Hotspot markers
    if (vizMode === 'hotspot' && heat_source_centroids && heat_source_centroids.length > 0) {
      const hPos = heat_source_centroids[0];
      if (probeMarkerRef.current) {
        probeMarkerRef.current.position.set(hPos[0], hPos[1], hPos[2]);
        probeMarkerRef.current.visible = true;
      }
    } else if (probeMarkerRef.current) {
      probeMarkerRef.current.visible = false;
    }

  }, [surfaceMesh, vizMode, tMin, tMax]);

  return (
    <div className="relative w-full h-full select-none overflow-hidden rounded-xl border border-graphite-800 bg-graphite-950">
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Engineering Temperature Legend */}
      <div className="absolute top-4 right-4 bg-graphite-900/90 backdrop-blur-md p-3.5 rounded-lg border border-graphite-700 shadow-xl flex flex-col items-center pointer-events-none">
        <span className="text-[10px] font-bold tracking-wider text-gray-400 mb-1">TEMPERATURE CONTOUR (°C)</span>
        <div className="flex items-center space-x-2">
          <span className="text-xs font-mono font-bold text-rose-500">{tMax.toFixed(1)}°</span>
          <div className="w-4 h-36 rounded-sm bg-gradient-to-t from-blue-600 via-cyan-400 via-yellow-400 to-red-600 border border-graphite-700 shadow-inner" />
          <span className="text-xs font-mono font-bold text-blue-400">{tMin.toFixed(1)}°</span>
        </div>
        <div className="mt-2 text-[9px] text-gray-500 uppercase tracking-widest font-mono">Finite Element Solution</div>
      </div>

      {/* Floating 3D Controls HUD */}
      <div className="absolute bottom-4 left-4 bg-graphite-900/80 backdrop-blur-md px-3 py-1.5 rounded-md border border-graphite-700 text-xs text-gray-300 flex items-center space-x-3 pointer-events-none">
        <span className="flex items-center space-x-1">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="font-mono text-cyan-300">Orbit: Left Click Drag</span>
        </span>
        <span className="text-gray-600">|</span>
        <span className="font-mono text-gray-400">Zoom: Mouse Wheel</span>
      </div>
    </div>
  );
};
