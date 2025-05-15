import { useCoverLetterInputStore } from "@/store/coverLetterStore";
import QuestionItem from "./QuestionItem";

function InputQuestion() {
  const { addQuestion, inputData } = useCoverLetterInputStore();
  const contentList = inputData.contents;
  const handleAddQuestion = () => {
    addQuestion();
  };

  console.log(contentList);
  return (
    <>
      {contentList.map((content, index) => (
        <QuestionItem content={content} index={index} />
      ))}
      <div
        onClick={handleAddQuestion}
        className="mt-2 py-3 px-5 border text-text-muted-foreground rounded-lg bg-background hover:bg-secondary-light hover:text-black hover:border hover:border-secondary"
      >
        + 문항 추가하기
      </div>
    </>
  );
}

export default InputQuestion;
