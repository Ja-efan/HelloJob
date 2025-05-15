import { Button } from "@/components/Button";
import { CoverLetterResponse } from "@/types/coverLetterTypes";
import { useState } from "react";

export interface CoverLetterEditor {
  selectQuestion: number;
  coverLetterId: number;
}

function CoverLetterEditor({
  selectQuestion,
  coverLetterId,
}: CoverLetterEditor) {
  console.log(selectQuestion, coverLetterId);
  //   const { isLoading, isError } = useGetCoverLetter(
  //     coverLetterId,
  //     selectQuestion
  //   );

  //   if (isLoading) {
  //     throw isLoading;
  //   }

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
  const [nowContentLength, setNowContentLength] = useState(
    data.content.contentDetail.length
  );
  const contentNumber = data.content.contentNumber;
  const contentDetail = data.content.contentDetail;
  const contentQuestion = data.content.contentQuestion;
  const contentLength = data.content.contentLength;

  const handleNowContentLength = (
    e: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setNowContentLength(e.target.textLength);
  };
  return (
    <div className="bg-white border w-[50rem] rounded-xl px-4 py-4">
      <div className="text-sm text-muted-foreground  pb-1">
        원하시는 부분을 수정하며 자소서를 완성해보세요!
      </div>
      <div className="text-2xl font-bold pb-2 ">
        {contentNumber}. {contentQuestion}
      </div>

      <label htmlFor="자기소개서 수정화면" />
      <textarea
        name="contentDetail"
        id="contentDetail"
        className="break-all w-full p-3 h-[68vh] border resize-none"
        onChange={handleNowContentLength}
        placeholder="자기소개서를 작성 해 주세요"
        maxLength={contentLength}
      >
        {contentDetail}
      </textarea>
      <div className="text-right text-muted-foreground">
        {nowContentLength} / {contentLength}
      </div>
      <div className="mt-4 flex justify-end gap-3">
        <Button variant={"white"} className="w-20">
          임시저장
        </Button>
        <Button className="w-20">저장</Button>
      </div>
    </div>
  );
}

export default CoverLetterEditor;
