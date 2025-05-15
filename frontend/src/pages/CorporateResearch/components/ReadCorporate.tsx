import { corporateReportApi } from "@/api/corporateReport";
import { useState, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FaRegBookmark, FaBookmark } from "react-icons/fa";
import { getCorporateReportDetailResponse } from "@/types/coporateResearch";
import { Button } from "@/components/Button";

interface ReadCorporateProps {
  id: number;
  onClose: () => void;
}

// 탭 타입 정의
type TabType = "공시 정보" | "뉴스 정보" | "분석 요약" | "최신 후기";

function ReadCorporate({ onClose, id }: ReadCorporateProps) {
  const queryClient = useQueryClient();
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [isValidId, setIsValidId] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>("공시 정보");

  // id 유효성 검사
  useEffect(() => {
    if (Number.isInteger(id) && id > 0) {
      setIsValidId(true);
    } else {
      setIsValidId(false);
    }
  }, [id]);

  // 기업 분석 레포트 상세 정보 불러오기
  const { data: reportDetail, isLoading } = useQuery({
    queryKey: ["corporateReportDetail", id],
    queryFn: async () => {
      const response = await corporateReportApi.getCorporateReportDetail(id);
      return response.data as getCorporateReportDetailResponse;
    },
    enabled: isValidId,
  });

  // reportDetail이 변경될 때 북마크 상태 업데이트
  useEffect(() => {
    if (reportDetail) {
      setIsBookmarked(reportDetail.bookmark);
    }
  }, [reportDetail]);

  // 북마크 추가 mutation
  const addBookmarkMutation = useMutation({
    mutationFn: () =>
      corporateReportApi.postBookmark({ companyAnalysisId: id }),
    onSuccess: () => {
      console.log("북마크 추가 성공");
      setIsBookmarked(true);
      queryClient.invalidateQueries({
        queryKey: ["corporateReportDetail", id],
      });
      queryClient.invalidateQueries({ queryKey: ["corporateReportList"] });
    },
  });

  // 북마크 삭제 mutation
  const removeBookmarkMutation = useMutation({
    mutationFn: () => corporateReportApi.deleteBookmark(id),
    onSuccess: () => {
      console.log("북마크 삭제 성공");
      setIsBookmarked(false);
      queryClient.invalidateQueries({
        queryKey: ["corporateReportDetail", id],
      });
      queryClient.invalidateQueries({ queryKey: ["corporateReportList"] });
    },
  });

  // 북마크 토글 핸들러
  const toggleBookmark = () => {
    if (isBookmarked) {
      removeBookmarkMutation.mutate();
    } else {
      addBookmarkMutation.mutate();
    }
  };

  return (
    <>
      <div className="h-[90vh] w-[940px] bg-white rounded-t-xl py-8 px-12 overflow-y-auto">
        <header className="flex w-full justify-between items-end mb-4">
          <h1 className="text-2xl font-bold">
            {isLoading || !reportDetail
              ? "로딩 중..."
              : reportDetail.companyName}{" "}
            기업 분석 -{" "}
            {isLoading || !reportDetail
              ? ""
              : new Date(reportDetail.createdAt).toLocaleDateString("ko-KR")}
          </h1>
          <button
            onClick={toggleBookmark}
            className="flex cursor-pointer items-center gap-1 px-3 py-2 rounded-md hover:bg-gray-100 transition-colors border border-gray-200"
          >
            {isBookmarked ? (
              <FaBookmark className="text-[#6F52E0]" />
            ) : (
              <FaRegBookmark />
            )}
            <span>북마크</span>
          </button>
        </header>

        {/* 탭 메뉴 */}
        <div className="border-b border-gray-200 mb-8">
          <div className="flex">
            {(
              ["공시 정보", "뉴스 정보", "분석 요약", "최신 후기"] as TabType[]
            ).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-2 px-4 text-center font-medium text-gray-700 hover:text-[#6F52E0] transition-colors ${
                  activeTab === tab
                    ? "text-[#6F52E0] border-b-2 border-[#6F52E0]"
                    : ""
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        {/* 탭 내용 */}
        {isLoading ? (
          <div className="flex justify-center items-center py-10">
            <p>로딩 중...</p>
          </div>
        ) : (
          <>
            {activeTab === "공시 정보" && (
              <PublicInformation reportDetail={reportDetail} />
            )}
            {activeTab === "뉴스 정보" && (
              <NewsInformation reportDetail={reportDetail} />
            )}
            {activeTab === "분석 요약" && (
              <AnalysisSummary reportDetail={reportDetail} />
            )}
            {activeTab === "최신 후기" && <LatestReviews />}
          </>
        )}
      </div>
      <div className="h-[80vh] mb-[10vh] w-[280px] flex flex-col justify-between bg-white rounded-xl border-t-6 border-[#AF9BFF] p-6">
        <header>
          <h1 className="text-xl font-bold">분석 결과 필터</h1>
        </header>
        <main></main>
        <footer className="flex gap-2 w-full justify-center">
          <Button className="px-6" variant={"white"}>
            이전
          </Button>
          <Button className="px-6" variant={"default"} onClick={onClose}>
            확인
          </Button>
        </footer>
      </div>
    </>
  );
}

// 공시 정보 컴포넌트
function PublicInformation({
  reportDetail,
}: {
  reportDetail?: getCorporateReportDetailResponse;
}) {
  if (!reportDetail) return null;

  return (
    <div className="space-y-12">
      <section>
        <h2 className="text-xl font-bold mb-6">기업 공시 기본 정보</h2>
        <div className="bg-gray-50 p-8 rounded-lg">
          <div className="space-y-4">
            <p>
              <span className="font-medium">설립일:</span> 2015년 3월 15일
            </p>
            <p>
              <span className="font-medium">직원 수:</span> 158명
            </p>
            <p>
              <span className="font-medium">산업 분야:</span> IT 서비스
            </p>
            <p>
              <span className="font-medium">매출액:</span> 187억 (2024년 기준)
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-xl font-bold mb-6">기업 공시 사업 계획</h2>
        <div className="bg-gray-50 p-8 rounded-lg">
          <div className="space-y-4">
            <p>
              삼성전자는 1969년에 설립되어 경기도 수원시에 본사를 두고 있는
              글로벌 전자 기업입니다. 대표이사는 전명헌이며, 약 12만 9천 명의
              직원을 보유하고 있습니다. 주요 사업 분야로는 소비자가전, 모바일,
              반도체, 디스플레이 네트워크 및 컴퓨팅 솔루션 등이 있습니다.
            </p>
            <p>
              2024년 삼성전자의 연간 매출은 약 300조 9천억 원으로, 전년 대비
              16.2% 증가하였으며, 영업이익은 약 32조 7천억 원으로
              집계되었습니다. 이는 2022년에 이어 역대 두 번째로 높은 매출을
              기록한 것입니다.
            </p>
            <p>
              인사 및 조직 측면에서, 2024년 삼성전자의 직원 평균 연봉은 약 1억
              2,800만 원으로 측정되며, 이는 전년 대비 약 7% 증가한 수치입니다.
              임원 평균 보수는 약 7억 2,600만 원으로 분석되었습니다. 또한,
              2024년에는 약 1만 960명을 신규로 채용하였으며, 퇴사자 수는 약
              6,459명으로 집계되었습니다.
            </p>
            <p>
              성과급 정책에 있어서, 삼성전자는 2024년부터 초과이익성과급(OPI)의
              일부를 자사주로 지급하기로 결정하였습니다. 이에 따라 상무는
              성과급의 50% 이상, 부사장은 70% 이상, 사장은 80% 이상을 자사주로
              수령해야 하며, 등기임원은 100% 자사주로 지급받습니다. 자사주는
              지급 후 일정 기간 동안 매도가 제한됩니다.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

// 뉴스 정보 컴포넌트
function NewsInformation({
  reportDetail,
}: {
  reportDetail?: getCorporateReportDetailResponse;
}) {
  if (!reportDetail) return null;

  return (
    <div>
      <h2 className="text-xl font-bold mb-6">뉴스 정보</h2>
      <div className="bg-gray-50 p-8 rounded-lg">
        <p className="mb-4">{reportDetail.newsAnalysisData}</p>
        <p className="text-sm text-gray-500 mb-6">
          분석 기준일: {reportDetail.newsAnalysisDate}
        </p>
        <h3 className="font-bold mb-3">참고 뉴스 URL</h3>
        <ul className="space-y-2">
          {reportDetail.newsAnalysisUrl.map((url, index) => (
            <li key={index}>
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#6F52E0] hover:underline"
              >
                {url}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// 분석 요약 컴포넌트
function AnalysisSummary({
  reportDetail,
}: {
  reportDetail?: getCorporateReportDetailResponse;
}) {
  if (!reportDetail) return null;

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-bold mb-6">분석 요약</h2>

      <section>
        <h3 className="font-bold text-lg mb-3">브랜드 정보</h3>
        <div className="bg-gray-50 p-6 rounded-lg">
          <p>{reportDetail.dartBrand}</p>
        </div>
      </section>

      <section>
        <h3 className="font-bold text-lg mb-3">현재 이슈</h3>
        <div className="bg-gray-50 p-6 rounded-lg">
          <p>{reportDetail.dartCurrIssue}</p>
        </div>
      </section>

      <section>
        <h3 className="font-bold text-lg mb-3">비전 및 전략</h3>
        <div className="bg-gray-50 p-6 rounded-lg">
          <p>{reportDetail.dartVision}</p>
        </div>
      </section>

      <section>
        <h3 className="font-bold text-lg mb-3">재무 요약</h3>
        <div className="bg-gray-50 p-6 rounded-lg">
          <p>{reportDetail.dartFinancialSummery}</p>
        </div>
      </section>

      <section>
        <h3 className="font-bold text-lg mb-3">카테고리</h3>
        <div className="flex flex-wrap gap-2">
          {reportDetail.dartCategory.map((category, index) => (
            <span
              key={index}
              className="px-3 py-1 bg-[#F0EBFF] text-[#6F52E0] rounded-full text-sm"
            >
              {category}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}

// 최신 후기 컴포넌트
function LatestReviews() {
  // 실제 데이터가 없으므로 예시 데이터 사용
  const reviews = [
    {
      id: 1,
      user: "사용자1",
      date: "2024-07-01",
      content:
        "기업 분석 보고서가 매우 상세하고 정확합니다. 투자 결정에 큰 도움이 되었습니다.",
    },
    {
      id: 2,
      user: "사용자2",
      date: "2024-06-28",
      content:
        "최근 이슈에 대한 분석이 잘 되어있어 기업의 방향성을 이해하는데 도움이 되었습니다.",
    },
    {
      id: 3,
      user: "사용자3",
      date: "2024-06-25",
      content:
        "재무 분석이 상세하게 되어있어 기업의 건전성을 파악하기 좋았습니다.",
    },
  ];

  return (
    <div>
      <h2 className="text-xl font-bold mb-6">최신 후기</h2>
      <div className="space-y-4">
        {reviews.map((review) => (
          <div key={review.id} className="border rounded-lg p-6 bg-gray-50">
            <div className="flex justify-between items-center mb-3">
              <p className="font-medium">{review.user}</p>
              <p className="text-sm text-gray-500">{review.date}</p>
            </div>
            <p>{review.content}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ReadCorporate;
