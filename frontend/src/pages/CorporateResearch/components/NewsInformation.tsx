import { getCorporateReportDetailResponse } from "@/types/coporateResearch";
import { useState } from "react";
import {
  FaNewspaper,
  FaCalendarAlt,
  FaLink,
  FaChevronDown,
  FaChevronUp,
} from "react-icons/fa";

// 뉴스 정보 컴포넌트
function NewsInformation({
  reportDetail,
}: {
  reportDetail?: getCorporateReportDetailResponse;
}) {
  const [showAllUrls, setShowAllUrls] = useState(false);

  if (!reportDetail) return null;

  // 뉴스 URL 처리
  const formattedUrls: string[] = reportDetail.newsAnalysisUrl || [];

  // 표시할 URL 개수 제한
  const displayUrls = showAllUrls ? formattedUrls : formattedUrls.slice(0, 3);

  return (
    <div className="space-y-10 pb-12">
      <section>
        <h3 className="font-bold text-2xl mb-4 text-[#2A2C35] flex items-center">
          <span className="bg-[#AF9BFF]/20 p-2 rounded-md mr-3 text-[#886BFB]">
            <FaNewspaper className="w-5 h-5" />
          </span>
          뉴스 분석 정보
        </h3>
        <div className="bg-white p-6 rounded-xl shadow-sm border border-[#E4E8F0] transition-all">
          <div className="mb-6">
            <div className="flex items-start space-x-3 mb-4">
              <div className="text-[#886BFB] mt-0.5">
                <FaCalendarAlt />
              </div>
              <div>
                <p className="text-sm text-[#6E7180]">뉴스 분석 날짜</p>
                <p className="font-semibold">{reportDetail.newsAnalysisDate}</p>
              </div>
            </div>

            <div className="flex items-start space-x-3">
              <div className="text-[#886BFB] mt-0.5 flex-shrink-0">
                <FaNewspaper />
              </div>
              <div>
                <p className="text-sm text-[#6E7180] mb-2">뉴스 분석 내용</p>
                <div className="leading-relaxed text-[#2A2C35] whitespace-pre-line">
                  {reportDetail.newsAnalysisData}
                </div>
              </div>
            </div>
          </div>

          <div className="border-t border-[#E4E8F0] pt-4">
            <div className="flex items-start">
              <div className="text-[#886BFB] mr-3 mt-0.5 flex-shrink-0">
                <FaLink />
              </div>
              <div className="w-full">
                <div className="flex justify-between items-center mb-2">
                  <p className="text-sm text-[#6E7180]">참고 뉴스 URL</p>
                  <button
                    onClick={() => setShowAllUrls(!showAllUrls)}
                    className="text-[#886BFB] cursor-pointer text-sm flex items-center hover:underline"
                  ></button>
                  {showAllUrls ? (
                    <button
                      onClick={() => setShowAllUrls(!showAllUrls)}
                      className="text-[#886BFB] cursor-pointer text-sm flex items-center hover:underline mr-5"
                    >
                      접기
                      <FaChevronUp className="ml-1" size={12} />
                    </button>
                  ) : (
                    <button
                      onClick={() => setShowAllUrls(!showAllUrls)}
                      className="text-[#886BFB] cursor-pointer text-sm flex items-center hover:underline"
                    >
                      모두 보기
                      <FaChevronDown className="ml-1" size={12} />
                    </button>
                  )}
                </div>
                <div className="space-y-2">
                  {displayUrls.map((url: string, index: number) => (
                    <a
                      key={index}
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-sm text-[#6F52E0] hover:underline truncate"
                    >
                      {url}
                    </a>
                  ))}
                  {!showAllUrls && formattedUrls.length > 3 && (
                    <p className="text-sm text-[#6E7180]">
                      외 {formattedUrls.length - 3}개 뉴스 (
                      {formattedUrls.length}개 중 {displayUrls.length}개 표시)
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default NewsInformation;
