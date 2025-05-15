import { Button } from "@/components/Button";

interface LetterStepProps {
  nowStep: number;
  handleStep: (stepNum: number) => void;
}

function LetterStep({ nowStep, handleStep }: LetterStepProps) {
  const stepTitle = [
    "기업/직무 입력",
    "1. 기업분석",
    "2. 직무분석",
    "3. 문항 입력",
    "4. 초안입력",
  ];

  const stepStyle = (select: boolean) => {
    return select
      ? "border flex justify-center items-center bg-primary text-white w-full h-15 text-lg font-semibold rounded-xl"
      : "border flex justify-center items-center w-full h-15 text-text-disabled text-lg font-semibold rounded-xl";
  };

  return (
    <aside className="sticky  bg-white top-20 border w-[18rem] h-full flex flex-col items-center gap-5 p-10 ">
      {stepTitle.map((title, index) => (
        <div
          key={index}
          className={stepStyle(nowStep === index)}
          onClick={() => handleStep(index)}
        >
          {title}
        </div>
      ))}
      <div className="flex gap-5">
        <Button variant={"white"} size="lg">
          이전
        </Button>
        <Button size="lg">다음</Button>
      </div>
    </aside>
  );
}

export default LetterStep;
