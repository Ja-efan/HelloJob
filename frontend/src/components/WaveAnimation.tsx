import React, { useEffect, useRef } from "react";

class Point {
  x: number;
  y: number;
  fixedY: number;
  speed: number;
  cur: number;
  max: number;

  constructor(index: number, x: number, y: number) {
    this.x = x;
    this.y = y;
    this.fixedY = y;
    this.speed = 0.03;
    this.cur = index;
    this.max = Math.random() * 100 + 20;
  }

  update() {
    this.cur += this.speed;
    this.y = this.fixedY + Math.sin(this.cur) * this.max;
  }
}

class Wave {
  index: number;
  totalPoints: number;
  color: string;
  points: Point[];
  stageWidth: number;
  stageHeight: number;
  centerX: number;
  centerY: number;
  pointGap: number;

  constructor(index: number, totalPoints: number, color: string) {
    this.index = index;
    this.totalPoints = totalPoints;
    this.color = color;
    this.points = [];
    this.stageWidth = 0;
    this.stageHeight = 0;
    this.centerX = 0;
    this.centerY = 0;
    this.pointGap = 0;
  }

  resize(stageWidth: number, stageHeight: number) {
    this.stageWidth = stageWidth;
    this.stageHeight = stageHeight;

    this.centerX = (stageWidth * 2) / 3;
    this.centerY = (stageHeight * 2) / 3;

    this.pointGap = this.stageWidth / (this.totalPoints - 1);

    this.init();
  }

  init() {
    this.points = [];

    for (let i = 0; i < this.totalPoints; i++) {
      const point = new Point(this.index + i, this.pointGap * i, this.centerY);
      this.points[i] = point;
    }
  }

  draw(ctx: CanvasRenderingContext2D) {
    ctx.beginPath();
    ctx.fillStyle = this.color;

    let prevX = this.points[0].x;
    let prevY = this.points[0].y;

    ctx.moveTo(prevX, prevY);

    for (let i = 1; i < this.totalPoints; i++) {
      if (i < this.totalPoints - 1) {
        this.points[i].update();
      }

      const cx = (prevX + this.points[i].x) / 2;
      const cy = (prevY + this.points[i].y) / 2;

      ctx.quadraticCurveTo(prevX, prevY, cx, cy);

      prevX = this.points[i].x;
      prevY = this.points[i].y;
    }

    ctx.lineTo(prevX, prevY);
    ctx.lineTo(this.stageWidth, this.stageHeight);
    ctx.lineTo(this.points[0].x, this.stageHeight);
    ctx.fill();
    ctx.closePath();
  }
}

class WaveGroup {
  totalWaves: number;
  totalPoints: number;
  color: string[];
  waves: Wave[];

  constructor() {
    this.totalWaves = 3;
    this.totalPoints = 6;

    this.color = [
      "rgba(74, 144, 250, 0.15)",
      "rgba(198, 70, 250, 0.15)",
      "rgba(136, 107, 251, 0.15)",
    ];

    this.waves = [];

    for (let i = 0; i < this.totalWaves; i++) {
      const wave = new Wave(i, this.totalPoints, this.color[i]);
      this.waves[i] = wave;
    }
  }

  resize(stageWidth: number, stageHeight: number) {
    for (let i = 0; i < this.totalWaves; i++) {
      const wave = this.waves[i];
      wave.resize(stageWidth, stageHeight);
    }
  }

  draw(ctx: CanvasRenderingContext2D) {
    for (let i = 0; i < this.totalWaves; i++) {
      const wave = this.waves[i];
      wave.draw(ctx);
    }
  }
}

interface WaveAnimationProps {
  className?: string;
}

const WaveAnimation: React.FC<WaveAnimationProps> = ({ className }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const waveGroupRef = useRef<WaveGroup | null>(null);
  const animationFrameRef = useRef<number>(0);
  const stageWidthRef = useRef<number>(0);
  const stageHeightRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // WaveGroup 인스턴스 생성
    waveGroupRef.current = new WaveGroup();

    // 리사이즈 함수
    const handleResize = () => {
      if (!canvas) return;

      stageWidthRef.current = window.innerWidth;
      stageHeightRef.current = window.innerHeight;

      canvas.width = stageWidthRef.current * 2;
      canvas.height = stageHeightRef.current * 2;
      ctx.scale(2, 2);

      if (waveGroupRef.current) {
        waveGroupRef.current.resize(
          stageWidthRef.current,
          stageHeightRef.current
        );
      }
    };

    // 애니메이션 함수
    const animate = () => {
      if (!ctx || !waveGroupRef.current) return;

      ctx.clearRect(0, 0, stageWidthRef.current, stageHeightRef.current);
      waveGroupRef.current.draw(ctx);
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    // 초기 설정
    window.addEventListener("resize", handleResize);
    handleResize();
    animate();

    // 정리 함수
    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute top-0 left-0 w-full h-full z-0 ${className || ""}`}
    />
  );
};

export default WaveAnimation;
