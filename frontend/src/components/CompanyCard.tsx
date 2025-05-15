import Avatar from "@/components/Avatar";
import React, { useEffect, useRef, useState } from "react";
import { FaClock, FaBuildingUser } from "react-icons/fa6";
import { cn } from "@/lib/utils";

function CardTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-title"
      className={cn("text-lg font-bold text-[#2A2C35]", className || "")}
      {...props}
    />
  );
}

function CardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-header"
      className={cn("flex flex-col items-center w-full", className || "")}
      {...props}
    />
  );
}

function CardFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-footer"
      className={cn(
        "flex justify-around text-[#6F52E0] w-full",
        className || ""
      )}
      {...props}
    />
  );
}

class GlowParticle {
  x: number;
  y: number;
  radius: number;
  rgb: { r: number; g: number; b: number };
  vx: number;
  vy: number;
  sinValue: number;

  constructor(
    x: number,
    y: number,
    radius: number,
    rgb: { r: number; g: number; b: number }
  ) {
    this.x = x;
    this.y = y;
    this.radius = radius;
    this.rgb = rgb;

    this.vx = Math.random() * 4;
    this.vy = Math.random() * 4;

    this.sinValue = Math.random();
  }

  animate(
    ctx: CanvasRenderingContext2D,
    stageWidth: number,
    stageHeight: number
  ) {
    this.sinValue += 0.01;

    this.radius += Math.sin(this.sinValue);

    this.x += this.vx;
    this.y += this.vy;

    if (this.x < 0) {
      this.vx *= -1;
      this.x += 10;
    } else if (this.x > stageWidth) {
      this.vx *= -1;
      this.x -= 10;
    }

    if (this.y < 0) {
      this.vy *= -1;
      this.y += 10;
    } else if (this.y > stageHeight) {
      this.vy *= -1;
      this.y -= 10;
    }

    ctx.beginPath();

    const g = ctx.createRadialGradient(
      this.x,
      this.y,
      this.radius * 0.01,
      this.x,
      this.y,
      this.radius
    );

    g.addColorStop(0, `rgba(${this.rgb.r}, ${this.rgb.g}, ${this.rgb.b}, 1)`);
    g.addColorStop(1, `rgba(${this.rgb.r}, ${this.rgb.g}, ${this.rgb.b}, 0)`);

    ctx.fillStyle = g;
    ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2, false);
    ctx.fill();
  }
}

interface CompanyCardProps {
  width: number;
  height: number;
  children?: React.ReactNode;
  className?: string;
  initialWidth?: number;
  initialHeight?: number;
  corName: string;
  corSize: string;
  industryName: string;
  region: string;
  updatedAt: string;
  onClick?: () => void;
  isGradient: boolean;
  colors?: { r: number; g: number; b: number }[];
}

// const DEFAULT_COLORS = [
//   { r: 45, g: 74, b: 227 },
//   { r: 255, g: 104, b: 248 },
//   { r: 54, g: 233, b: 84 },
// ];

// const COLORS = [
//   { r: 106, g: 44, b: 112 },
//   { r: 249, g: 237, b: 105 },
//   { r: 240, g: 138, b: 93 },
//   { r: 184, g: 59, b: 94 },
// ];
const DEFAULT_COLORS = [
  { r: 117, g: 185, b: 190 },
  { r: 222, g: 145, b: 81 },
  // { r: 179, g: 57, b: 81 },
  { r: 0, g: 187, b: 167 },
  { r: 136, g: 107, b: 251 },
];

const CompanyCard: React.FC<CompanyCardProps> = ({
  width,
  height,
  children,
  className,
  corName,
  corSize,
  industryName,
  initialWidth = 230,
  initialHeight = 180,
  isGradient = false,
  updatedAt,
  onClick,
  colors = DEFAULT_COLORS,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<GlowParticle[]>([]);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    if (!isHovered) return;
    if (!isGradient) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pixelRatio = window.devicePixelRatio > 1 ? 2 : 1;
    const totalParticles = 12;
    const maxRadius = 360;
    const minRadius = 160;

    // 캔버스 설정
    canvas.width = width * pixelRatio;
    canvas.height = height * pixelRatio;
    ctx.scale(pixelRatio, pixelRatio);

    ctx.globalCompositeOperation = "source-over";

    // 파티클 생성
    let curColor = 0;
    const particles: GlowParticle[] = [];

    for (let i = 0; i < totalParticles; i++) {
      const item = new GlowParticle(
        Math.random() * width,
        Math.random() * height,
        Math.random() * (maxRadius - minRadius) + minRadius,
        colors[curColor]
      );

      if (++curColor >= colors.length) {
        curColor = 0;
      }

      particles[i] = item;
    }

    particlesRef.current = particles;

    // 애니메이션 함수
    const animate = () => {
      ctx.clearRect(0, 0, width, height);

      for (let i = 0; i < totalParticles; i++) {
        const item = particles[i];
        item.animate(ctx, width, height);
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    // 컴포넌트 언마운트 시 애니메이션 정리
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [width, height, isHovered, colors]);

  return (
    <div
      className={`relative transition-all duration-150 ease-in-out`}
      style={{
        width: isHovered && isGradient ? width : initialWidth,
        height: isHovered && isGradient ? height : initialHeight,
        cursor: "pointer",
        zIndex: isHovered && isGradient ? 31 : 1,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {isHovered && isGradient ? (
        <>
          <canvas
            ref={canvasRef}
            className={`w-full h-full absolute top-0 left-0 rounded-lg ${className} border border-[#6F4BFF]`}
            style={{
              zIndex: isHovered ? 30 : 0,
            }}
          />
          <div
            className="absolute w-full h-full"
            onClick={onClick}
            style={{ zIndex: isHovered ? 50 : 20 }}
          >
            {children}
          </div>
        </>
      ) : (
        <div
          className={`w-full h-full rounded-lg ${className}`}
          onClick={onClick}
        >
          <div className="w-full h-full bg-white flex flex-col justify-between p-4 items-center rounded-lg border border-[#AF9BFF] border-t-4 shadow-sm">
            <CardHeader>
              <Avatar username={corName} size={42} className="bg-[#F1F3F9]" />
              <CardTitle className="pt-2">{corName}</CardTitle>
              <span className="text-[#6E7180] text-sm">{industryName}</span>
            </CardHeader>
            <CardFooter>
              <div className="flex gap-1 items-center">
                <FaBuildingUser className="w-5" />
                <span className="text-sm">{corSize}</span>
              </div>
              <div className="flex gap-1 items-center">
                <FaClock className="w-5" />
                <span className="text-sm">{updatedAt}</span>
              </div>
            </CardFooter>
          </div>
        </div>
      )}
    </div>
  );
};

export default CompanyCard;
