import { ReactNode } from "react";
import LetterStep from "./LetterStep";
import { useSelectCompanyStore } from "@/store/coverLetterAnalysisStore";

interface CoverLetterAnalysisLayoutProps {
  nowStep: number;
  children: ReactNode;
  handleStep: (nowStep: number) => void;
}

function CoverLetterAnalysisLayout({
  children,
  nowStep,
  handleStep,
}: CoverLetterAnalysisLayoutProps) {
  const { company } = useSelectCompanyStore();
  const title = [
    "기업/직무 선택",
    "기업분석 선택하기",
    "직무분석 선택하기",
    "자기소개서 문항 입력",
  ];

  const subTitle = [
    <>
      <p className="mb-2">지원하실 기업/직무를 선택해 주세요</p>
      <p className="text-text-muted-foreground text-lg"></p>
    </>,
    <>
      <p className="mb-2">
        {company.companyName}의 기업분석 자료를 선택해 주세요
      </p>
      <p className="text-text-muted-foreground text-lg">
        즐겨찾기해놓은 기업분석 자료가 없다면 추가해 보세요!
      </p>
    </>,
    <>
      <p className="mb-2">
        {company.companyName}의 직무분석 자료를 선택해 주세요
      </p>
      <p className="text-text-muted-foreground text-lg">
        즐겨찾기해놓은 직무분석 자료가 없다면 추가해 보세요!
      </p>
    </>,
    <>
      <p className="mb-2">문항을 입력해 주세요</p>
      <p className="text-text-muted-foreground text-lg">
        자소서 문항을 추가하고 포트폴리오, 경험을 원하는 문항에 연결해 보세요!
      </p>
    </>,
  ];
  return (
    <>
      <div className="flex justify-center gap-10 mt-10 mb-20">
        <div className="bg-white border w-full border-t-4 border-t-primary rounded-xl p-10 ">
          <div className="text-2xl font-bold pb-3 border-b-1">
            {title[nowStep]}
          </div>
          <div className="text-3xl font-semibold my-10">
            {subTitle[nowStep]}
          </div>
          <div className="mt-10 mb-20">{children}</div>
        </div>
        <LetterStep nowStep={nowStep} handleStep={handleStep} />
      </div>
    </>
  );
}

export default CoverLetterAnalysisLayout;
