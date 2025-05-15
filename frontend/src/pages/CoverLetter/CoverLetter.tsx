import { useCallback, useMemo, useRef, useState } from "react";
import { CoverLetterResponse } from "@/types/coverLetterTypes";
import InputChat from "./components/InputChat";
import QuestionStep from "./components/QuestionStep";
import { useCoverLetterStore } from "@/store/coverLetterStore";
import CoverLetterEditor from "./components/CoverLetterEditor";
import { useParams } from "react-router";

function CoverLetter() {
  const data: CoverLetterResponse = {
    coverLetterId: 1, // 자기소개서 id
    summary: {
      totalContentQuestionCount: 4, // 총 문항 수
      contentQuestionStatuses: [
        // 문항별 작성 상태(예: 1번 문항 - 작성 완료)
        { contentNumber: 1, contentStatus: "COMPLETED" },
        { contentNumber: 2, contentStatus: "IN_PROGRESS" },
        { contentNumber: 3, contentStatus: "PENDING" },
        { contentNumber: 4, contentStatus: "PENDING" },
      ],
      companyAnalysisId: 1, // 기업 분석 id
      jobRoleSnapshotId: 1, // 직무 분석 id(null일 수도 있음)
      coverLetterUpdatedAt: "2025-04-24T13:03:00", // 전체 자소서 수정일
    },
    content: {
      contentQuestion: "지원 동기를 적어주세요.", // 자기소개서 질문
      contentNumber: 1, // 자기소개서 문항 번호
      contentLength: 700, // 글자수 제한
      contentDetail: "안녕하세요, 저는 프론트엔드 개발자 지망생입니다.....",
      contentExperienceIds: [1, 2], // 선택한 경험 id
      contentProjectIds: [3], // 선택한 프로젝트 id
      contentFirstPrompt: "이거 이렇게 저렇게 요렇게 하고 싶음", // 작성 요청 시의 처음 프롬프트
      contentStatus: "IN_PROGRESS", // "PENDING": 미작성 | "IN_PROGRESS": 작성 중 | "COMPLETED": 작성 완료,
      contentChatLog: [
        { sender: "USER", message: "유저입니다" },
        { sender: "AI", message: "이렇게 바꿔보세요?" },
      ],
      contentUpdatedAt: "2025-04-23T08:50:37.000", // 개별 자소서 문항의 수정일
    },
  };
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const [selectQuestion, setSelectQuestion] = useState(1);
  const param = useParams();
  const { chatLog } = useCoverLetterStore();

  const coverLetterId = param.id ? parseInt(param.id, 10) : 0;

  const handleSelectQuestion = (selectNum: number) => {
    setSelectQuestion(selectNum);
  };

  const onSubmitMessage = useCallback(() => {
    const chatContainer = chatContainerRef.current;
    setTimeout(() => {
      if (chatContainer) {
        chatContainer.scrollTo({
          top: chatContainer.scrollHeight,
          behavior: "smooth",
        });
      }
    }, 0);
  }, []);

  const QuestionStatuses = data.summary.contentQuestionStatuses;

  const chatStyles = useMemo(
    () => ({
      USER: {
        container: "flex flex-row-reverse",
        chatBubble:
          "break-all border max-w-92 px-3 py-3 rounded-xl bg-primary text-white",
      },
      AI: {
        container: "flex",
        chatBubble: "break-all border max-w-92 px-3 py-3 rounded-xl bg-muted",
      },
    }),
    []
  );

  return (
    <>
      <div className="flex mt-5 gap-3 items-start">
        <div className="bg-white border w-[50rem] border-t-4 border-t-primary rounded-xl px-4 py-4 ">
          <div className="text-2xl font-bold pb-1 ">첨삭 도우미</div>
          <div className="text-sm text-muted-foreground border-b-1 pb-1">
            원하시는 부분을 수정하며 자소서를 완성해보세요!
          </div>

          <div className="relative">
            <div
              ref={chatContainerRef}
              className="flex flex-col h-[76vh] grow overflow-y-auto gap-2 mt-2 pb-15"
            >
              {chatLog.map((chat, index) => (
                <div key={index} className={chatStyles[chat.sender].container}>
                  <div className={chatStyles[chat.sender].chatBubble}>
                    {chat.message}
                  </div>
                </div>
              ))}
            </div>
            <div className="absolute bottom-0 w-full">
              <InputChat onSubmitMessage={onSubmitMessage}></InputChat>
            </div>
          </div>
        </div>

        <CoverLetterEditor
          selectQuestion={selectQuestion}
          coverLetterId={coverLetterId}
        />

        <QuestionStep
          QuestionStatuses={QuestionStatuses}
          handleSelectQuestion={handleSelectQuestion}
          selectQuestion={selectQuestion}
        />
      </div>
    </>
  );
}

export default CoverLetter;
