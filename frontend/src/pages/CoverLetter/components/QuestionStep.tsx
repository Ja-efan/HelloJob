import { QuestionStepProps } from "@/types/coverLetterTypes";

function QuestionStep({
  QuestionStatuses,
  handleSelectQuestion,
  selectQuestion,
}: QuestionStepProps) {
  const questions = [1, 2, 3];

  const stepStatusStyle = (questionStatus: string) => {
    switch (questionStatus) {
      case "COMPLETED":
        return "bg-completed";
      case "IN_PROGRESS":
        return "bg-in-progress";
      case "PENDING":
        return "bg-pending";
    }
  };

  const stepBlockStyle = (select: boolean) => {
    return select
      ? "border flex justify-center items-center bg-primary text-white w-15 aspect-square text-lg font-semibold rounded-xl"
      : "border flex justify-center items-center w-15 bg-white aspect-square text-text-disabled text-lg font-semibold rounded-xl text-black";
  };
  return (
    <div className="flex flex-col gap-3  justify-center items-center">
      {questions.map((num, index) => (
        <div
          key={index}
          onClick={() => handleSelectQuestion(num)}
          className={`flex flex-col ${stepBlockStyle(num === selectQuestion)}`}
        >
          <div>{num}</div>
          <div
            className={`size-2 rounded-full ${stepStatusStyle(
              QuestionStatuses[num].contentStatus
            )}`}
          ></div>
        </div>
      ))}
    </div>
  );
}

export default QuestionStep;
