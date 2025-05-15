import Avatar from "@/components/Avatar";
import CompanyCard from "@/components/CompanyCard";

const PALETTE = [
  { r: 236, g: 0, b: 63 }, // rose-600
  { r: 230, g: 0, b: 118 }, // pink-600
  { r: 200, g: 0, b: 222 }, // fuchsia-600
  { r: 152, g: 16, b: 250 }, // purple-600
  { r: 21, g: 93, b: 252 }, // blue-600
  { r: 0, g: 132, b: 209 }, // sky-600
  { r: 0, g: 146, b: 184 }, // cyan-600
  { r: 0, g: 150, b: 137 }, // teal-600
  { r: 0, g: 153, b: 102 }, // emerald-600
  { r: 94, g: 165, b: 0 }, // lime-600
  { r: 208, g: 135, b: 0 }, // yellow-600
  { r: 225, g: 113, b: 0 }, // amber-600
  { r: 231, g: 0, b: 11 }, // red-600
];

function stringToPaletteClass(str: string): {
  r: number;
  g: number;
  b: number;
} {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    // Simple hash function (djb2‑ish)
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const index = Math.abs(hash) % PALETTE.length;
  return PALETTE[index];
}

interface GradientCardProps {
  id: string;
  width: number;
  height: number;
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
}

const GradientCard: React.FC<GradientCardProps> = ({
  id,
  width,
  height,
  className,
  corName,
  corSize,
  industryName,
  initialWidth = 230,
  initialHeight = 180,
  region,
  updatedAt,
  onClick,
  isGradient = false,
}) => {
  const colors = [];
  if (corSize === "대기업") {
    colors.push({ r: 45, g: 74, b: 227 });
  } else if (corSize === "중견기업") {
    colors.push({ r: 54, g: 233, b: 84 });
  } else if (corSize === "중소기업") {
    colors.push({ r: 255, g: 104, b: 248 });
  }
  //가장 색이 많이 노출되는 4번째 자리에 메인 색을 마지막에 넣어준다
  colors.push(stringToPaletteClass(corName));
  colors.push({ r: 136, g: 107, b: 251 });

  return (
    <div
      className="relative overflow-visible"
      key={id}
      style={{ width: `${initialWidth}px`, height: `${initialHeight}px` }}
    >
      <CompanyCard
        width={width}
        height={height}
        className={className}
        initialWidth={initialWidth}
        initialHeight={initialHeight}
        corName={corName}
        corSize={corSize}
        industryName={industryName}
        isGradient={isGradient}
        colors={colors}
        region={region}
        updatedAt={updatedAt}
        onClick={onClick}
      >
        <div className="w-full h-full bg-black/10 px-4 py-8 rounded-lg flex flex-col items-center">
          <Avatar username={corName} size={50} className="bg-[#F1F3F9]" />
          <h2 className="text-white text-xl font-bold pt-4 pb-4">{corName}</h2>
          <div className="flex flex-col gap-2 w-full">
            <div>
              <p className="text-gray-100 text-xs">지역</p>
              <p className="text-white font-semibold">{region}</p>
            </div>
            <div>
              <p className="text-gray-100 text-xs">업종명</p>
              <p className="text-white font-semibold">{industryName}</p>
            </div>
            <div>
              <p className="text-gray-100 text-xs">기업 규모</p>
              <p className="text-white font-semibold">{corSize}</p>
            </div>
            <div>
              <p className="text-gray-100 text-xs">최근 레포트 등록일</p>
              <p className="text-white font-semibold">{updatedAt}</p>
            </div>
          </div>
        </div>
      </CompanyCard>
    </div>
  );
};

export default GradientCard;
