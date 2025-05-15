import FormInput from "@/components/Common/FormInput";
import { CoverLetterRequestContent } from "@/types/coverLetterTypes";
import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import ProjectForm from "@/pages/Resume/ProjectForm";

export interface QuestionItemProps {
  index: number;
  content: CoverLetterRequestContent;
}

function QuestionItem({ content }: QuestionItemProps) {
  const headerStyle =
    "w-full text-primary bg-secondary-light rounded-t-2xl py-3 px-4 font-semibold";

  // 문항 열림/닫힘 상태 관리
  const [isOpen, setIsOpen] = useState(true);
  const [charCount, setCharCount] = useState(0);
  const [projectFormOpen, setProjectFormOpen] = useState(false);
  // 토글 핸들러
  const toggleForm = () => {
    setIsOpen(!isOpen);
  };

  const handleProjectFormClose = () => {
    setProjectFormOpen(false);
  };

  const handleProjectFormOpen = () => {
    setProjectFormOpen(true);
  };

  return (
    <>
      {/* <ProjectModal /> */}
      {projectFormOpen && <ProjectForm onClose={handleProjectFormClose} />}
      <form action="" className="border w-full rounded-2xl mb-3">
        <div
          className={`${headerStyle} flex justify-between items-center cursor-pointer`}
          onClick={toggleForm}
        >
          <span>{content.contentNumber}번 문항</span>
          {isOpen ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>

        {isOpen && (
          <div className="mx-4 flex flex-col gap-3 py-3">
            <div className="flex gap-3 mt-3 ">
              <div className="w-full">
                <FormInput
                  type="text"
                  name="contentQuestion"
                  className="border w-full"
                  placeholder="문항을 입력해 주세요"
                  width={""}
                  height={""}
                />
              </div>
              <div>
                <FormInput
                  type="number"
                  name="contentLength"
                  placeholder="글자수"
                  className="w-30"
                  width={""}
                  height={""}
                />
              </div>
            </div>
            <div className="flex gap-8">
              <div className="w-[50%]">
                <div className={`${headerStyle}`}>프로젝트</div>
                <div className="flex flex-col gap-2.5 border p-3 rounded-b-xl">
                  <div className="cursor-pointer border border-l-4 border-l-accent py-3 truncate px-5 rounded-lg">
                    Hello Job 프로젝트
                  </div>
                  <div
                    onClick={handleProjectFormOpen}
                    className="border border-transparent py-3 px-5 text-text-muted-foreground rounded-lg bg-background hover:bg-secondary-light hover:text-black hover:border hover:border-secondary"
                  >
                    + 프로젝트 연결하기
                  </div>
                </div>
              </div>

              <div className="w-[50%]">
                <div className={`${headerStyle}`}>경험</div>
                <div className="flex flex-col gap-2.5 border p-3 rounded-b-xl">
                  <div className="border border-transparent py-3 px-5 text-text-muted-foreground rounded-lg bg-background hover:bg-secondary-light hover:text-black hover:border hover:border-secondary">
                    + 경험 연결하기
                  </div>
                </div>
              </div>
            </div>
            <div className="relative">
              <div className={headerStyle}>내용 추가</div>
              <textarea
                name="contentFirstPrompt"
                placeholder="추가하고 싶은 내용들을 적어주세요!"
                maxLength={1500}
                rows={8}
                cols={50}
                className="bg-white resize-none border rounded-b-xl w-full p-4 pb-10 "
                onChange={(e) => setCharCount(e.target.value.length)}
              />
              <span className="bg-white rounded-2xl px-2 absolute mt-10 right-4 bottom-3 text-sm text-text-muted-foreground">
                {charCount} / 1500
              </span>
            </div>
          </div>
        )}
      </form>
    </>
  );
}

export default QuestionItem;
