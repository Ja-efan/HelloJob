import { FadeLoader } from "react-spinners";

interface LoadingProps {
  className?: string;
  radius?: number;
}
function Loading({ className, radius = 5 }: LoadingProps) {
  return (
    <div className={`flex justify-center items-center  ${className}`}>
      <FadeLoader radius={radius} color="#886bfb" />
    </div>
  );
}

export default Loading;
