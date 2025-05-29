import { getCorporateReportDetailResponse } from "@/types/coporateResearch";
import { FaThLarge, FaBookReader } from "react-icons/fa";
import SWOTCard from "./SWOTCard";

function SWOTAnalysis({
  reportDetail,
}: {
  reportDetail?: getCorporateReportDetailResponse;
}) {
  if (!reportDetail) return null;

  const swotStrengthContent = reportDetail.swotStrengthContent;
  const swotWeaknessContent = reportDetail.swotWeaknessContent;
  const swotOpportunityContent = reportDetail.swotOpportunityContent;
  const swotThreatContent = reportDetail.swotThreatContent;
  const swotStrengthTag = reportDetail.swotStrengthTag;
  const swotWeaknessTag = reportDetail.swotWeaknessTag;
  const swotOpportunityTag = reportDetail.swotOpportunityTag;
  const swotThreatTag = reportDetail.swotThreatTag;

  // SWOT 관련 내용이 모두 없는 경우 체크
  const hasNoSWOTContent =
    !swotStrengthContent?.length &&
    !swotWeaknessContent?.length &&
    !swotOpportunityContent?.length &&
    !swotThreatContent?.length;

  return (
    <div className="space-y-10 pb-12">
      <section>
        <h3 className="font-bold text-2xl mb-4 text-[#2A2C35] flex items-center">
          <span className="bg-[#AF9BFF]/20 p-2 rounded-md mr-3 text-[#886BFB]">
            <FaThLarge className="w-5 h-5" />
          </span>
          SWOT 분석
        </h3>
        {hasNoSWOTContent ? (
          <div className="p-6 text-center text-gray-600 bg-gray-50 rounded-lg">
            SWOT 분석을 포함하지 않는 기업 분석입니다.
          </div>
        ) : (
          <div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Strength */}
              <SWOTCard
                title="Strength (강점)"
                items={swotStrengthContent}
                tags={swotStrengthTag}
                color="bg-green-50"
                accent="text-green-700"
                tagColor="bg-green-200"
              />
              <SWOTCard
                title="Weakness (약점)"
                items={swotWeaknessContent}
                tags={swotWeaknessTag}
                color="bg-red-50"
                accent="text-red-700"
                tagColor="bg-red-200"
              />
              <SWOTCard
                title="Opportunity (기회)"
                items={swotOpportunityContent}
                tags={swotOpportunityTag}
                color="bg-yellow-50"
                accent="text-yellow-700"
                tagColor="bg-yellow-200"
              />
              <SWOTCard
                title="Threat (위협)"
                items={swotThreatContent}
                tags={swotThreatTag}
                color="bg-indigo-50"
                accent="text-indigo-700"
                tagColor="bg-indigo-200"
              />
            </div>
            {/* 종합 분석 */}
            <div className="mt-6">
              <h2 className="font-bold text-xl text-[#2A2C35] flex items-center">
                <span className="text-[#886BFB] px-1 mr-2">
                  <FaBookReader className="w-5 h-5" />
                </span>
                종합 분석
              </h2>
              <div className="mt-4 p-5 rounded-xl shadow-sm bg-stone-100 transition-all">
                <p className="text-base whitespace-pre-wrap text-[#2A2C35]">
                  {reportDetail.swotSummary}
                </p>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default SWOTAnalysis;
