import { useState } from "react";
import { AiOutlineSearch } from "react-icons/ai";
import { useSelectCompanyStore } from "@/store/coverLetterAnalysisStore";
import JobSearch from "../SearchInputModal/JobSearch";
import CompanySearch from "../SearchInputModal/CompanySearch";

export type SearchType = "job" | "company" | "";

function JobCompanyForm() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const { company } = useSelectCompanyStore();
  const modalClose = () => {
    setIsModalOpen(false);
  };

  const modalOpen = () => {
    setIsModalOpen(true);
  };

  const handleCompanyClick = () => {
    modalOpen();
  };

  return (
    <>
      {isModalOpen && <CompanySearch modalClose={modalClose}></CompanySearch>}
      <div className="flex w-full h-auto  justify-center  gap-5 ">
        <div className="border rounded-xl w-full p-5 ">
          <div className="font-semibold  text-xl">기업 선택</div>
          {company.companyId !== 0 ? (
            <div
              onClick={handleCompanyClick}
              className="mt-2 cursor-pointer text-sm border rounded-xl grid grid-cols-10 items-center h-11 px-3 hover:border-primary hover:border-2 transition-all duration-150 hover:bg-secondary-light group:active:bg-primary  "
            >
              <span className=" col-span-1 group-active:bg-active group-active:border-active duration-100 border bg-secondary-light rounded-full size-7 flex justify-center items-center font-semibold text-primary text-sm aspect-square">
                {company.companyName[0]}
              </span>
              <span className="col-span-3 font-medium text-left ml-3">
                {company.companyName}
              </span>
              <span className="col-span-3 text-right ">
                {company.companyLocation}
              </span>
              <span className="col-span-3 text-right ">
                {company.companySize}
              </span>
            </div>
          ) : (
            <>
              <div
                onClick={handleCompanyClick}
                className="bg-background font-semibold text-text-muted-foreground flex justify-center items-center gap-2 border rounded h-10 mt-3 w-full"
              >
                <div>기업명을 검색하세요</div>
                <AiOutlineSearch />
              </div>
              <div className="mt-3 *:font-semibold text-text-muted-foreground">
                최근 검색 기업
              </div>
            </>
          )}
        </div>
        <JobSearch></JobSearch>
      </div>
    </>
  );
}
export default JobCompanyForm;

// enum("서버/백엔드 개발자", "프론트엔드 개발자", "웹 풀스택 개발자", "안드로이드 개발자", "iOS 개발자", "크로스플랫폼 앱개발자", "게임 클라이언트 개발자", "게임 서버 개발자", "DBA", "빅데이터 엔지니어", "인공지능/머신러닝", "devops/시스템 엔지니어", "정보보안 담당자", "QA 엔지니어", "개발 PM", "HW/임베디드", "SW/솔루션", "웹퍼블리셔", "VR/AR/3D", "블록체인", "기술지원", "기타")
