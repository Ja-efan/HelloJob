import { Button } from "@/components/Button";
import { useNavigate, useParams } from "react-router";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { corporateReportApi } from "@/api/corporateReport";

import { FaPlus } from "react-icons/fa6";
import DetailModal from "@/components/Common/DetailModal";
import CreateCorporate from "./components/CreateCorporate";
import ReadCorporate from "./components/ReadCorporate";
import { getCorporateReportListResponse } from "@/types/coporateResearch";
import CorporateReportCard from "./components/CorporateReportCard";

interface CorporateReport {
  companyAnlaysisId: number;
  companyName: string;
  createdAt: string;
  companyViewCount: number;
  companyAnalysisBookmarkCount: number;
  companyLocation: string;
  bookmark: boolean;
  dartCategory: string[];
  public: boolean;
}

function CorporateResearch() {
  const params = useParams();
  const navigate = useNavigate();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalView, setModalView] = useState<"create" | "read">("create");
  const [researchId, setResearchId] = useState<number>(1);

  // tanstack query를 사용한 특정 기업의 모든 리포트 불러오기
  const { data: corporateReportListData, isLoading } = useQuery({
    queryKey: ["corporateReportList", params.id],
    queryFn: async () => {
      const response = await corporateReportApi.getCorporateReportList(
        parseInt(params.id ? params.id : "1")
      );
      return response.data as getCorporateReportListResponse[];
    },
  });

  const [corporateReportList, setCorporateReportList] = useState<
    CorporateReport[]
  >([]);

  useEffect(() => {
    const temp =
      corporateReportListData?.map((corporateReport) => ({
        companyAnlaysisId: corporateReport.companyAnlaysisId,
        companyName: corporateReport.companyName,
        createdAt: corporateReport.createdAt,
        companyViewCount: corporateReport.companyViewCount,
        companyLocation: corporateReport.companyLocation,
        companyAnalysisBookmarkCount:
          corporateReport.companyAnalysisBookmarkCount,
        bookmark: corporateReport.bookmark,
        dartCategory: corporateReport.dartCategory,
        public: corporateReport.public,
      })) || [];
    setCorporateReportList(temp);
    debugger;
  }, [corporateReportListData]);

  const openCreateModal = () => {
    setModalView("create");
    setIsModalOpen(true);
  };

  const openReadModal = (id: number) => {
    setResearchId(id);
    setModalView("read");
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  return (
    <div className="flex flex-col justify-between w-full h-full p-6">
      <h2 className="text-2xl font-bold mb-4">기업 분석 검색 결과</h2>
      <h1 className="text-3xl font-bold mb-1">삼성 전자</h1>
      <h1 className="text-3xl font-bold mb-12">기업 분석 레포트 목록입니다</h1>
      <div className="flex justify-start gap-4 w-[1114px] mx-auto flex-wrap">
        <button className="cursor-pointer" onClick={openCreateModal}>
          <div className="w-[210px] h-[180px] rounded-lg group border border-dashed border-[#886BFB] flex flex-col items-center justify-center gap-2 hover:border-[#6F52E0] transition-colors">
            <div className="w-8 h-8 flex items-center justify-center rounded-full bg-[#AF9BFF] group-hover:bg-[#886BFB] transition-colors text-white">
              <FaPlus />
            </div>
            <span className="text-[#6E7180] group-hover:text-black transition-colors">
              기업 분석 추가하기
            </span>
          </div>
        </button>
        {isLoading ? (
          <div>로딩 중...</div>
        ) : corporateReportList.length > 0 ? (
          corporateReportList.map((corporateReport) => (
            <CorporateReportCard
              key={corporateReport.companyAnlaysisId}
              onClick={() => {
                openReadModal(corporateReport.companyAnlaysisId);
              }}
              companyName={corporateReport.companyName}
              createdAt={corporateReport.createdAt}
              companyViewCount={corporateReport.companyViewCount}
              companyLocation={corporateReport.companyLocation}
              companyAnalysisBookmarkCount={
                corporateReport.companyAnalysisBookmarkCount
              }
              bookmark={corporateReport.bookmark}
              dartCategory={corporateReport.dartCategory}
            />
          ))
        ) : (
          <div className="text-center text-gray-500 text-lg h-[180px] w-[210px] flex flex-col items-center justify-center">
            <h1>검색 결과가 없습니다.</h1>
            <h1 className="text-base mt-2">기업 분석을 진행해주세요.</h1>
          </div>
        )}
      </div>

      <footer className="fixed left-0 bottom-0 w-full flex justify-center gap-4 pb-6 pt-10 bg-gradient-to-t from-[#FFFFFF]/70 via-[#FFFFFF]/70 to-transparent">
        <Button
          onClick={() => navigate(-1)}
          variant={"white"}
          className="text-base"
        >
          이전
        </Button>
        <Button
          onClick={() => navigate(`/job-research/${params.id}`)}
          variant={"default"}
          className="text-base"
        >
          직무 분석으로
        </Button>
      </footer>

      {isModalOpen && (
        <DetailModal isOpen={isModalOpen} onClose={closeModal}>
          {modalView === "create" ? (
            <CreateCorporate onClose={closeModal} />
          ) : (
            <ReadCorporate onClose={closeModal} id={researchId} />
          )}
        </DetailModal>
      )}
    </div>
  );
}

export default CorporateResearch;
