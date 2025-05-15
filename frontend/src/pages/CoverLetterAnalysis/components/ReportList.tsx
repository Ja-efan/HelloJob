import { useGetCompanyBookMarks } from "@/hooks/companyHooks";
import { useGetJobBookMarks } from "@/hooks/jobRoleAnalysisHook";
import { useSelectCompanyStore } from "@/store/coverLetterAnalysisStore";
import { useCoverLetterInputStore } from "@/store/coverLetterStore";
import {
  CompanyBookMarkResponse,
  JobBookMarkResponse,
  ReportListProps,
} from "@/types/coverLetterTypes";

function ReportList({ nowStep }: ReportListProps) {
  const reportBlockLayout =
    "p-5 border w-full rounded-2xl h-35 flex justify-center items-center cursor-pointer ";
  const hoverReportBlockLayout =
    "duration-100 hover:border-2 hover:border-primary hover:bg-secondary-light";
  const selectedStyle = "border-2 border-primary bg-secondary-light ";

  const { company } = useSelectCompanyStore();
  const { setCompanyAnalysisId, setJobRoleAnalysisId, inputData } =
    useCoverLetterInputStore();
  const companyBookMarksQuery = useGetCompanyBookMarks(company.companyId);
  const jobBookMarksQuery = useGetJobBookMarks(company.companyId);

  const data =
    nowStep === 1 ? companyBookMarksQuery.data : jobBookMarksQuery.data;

  const handleSelect = (analysisId: number) => {
    if (nowStep === 1) {
      inputData.companyAnalysisId === analysisId
        ? setCompanyAnalysisId(null)
        : setCompanyAnalysisId(analysisId);
    } else if (nowStep === 2) {
      inputData.jobRoleAnalysisId === analysisId
        ? setJobRoleAnalysisId(null)
        : setJobRoleAnalysisId(analysisId);
    }
  };

  function adaptData(
    data: JobBookMarkResponse[] | CompanyBookMarkResponse[],
    nowStep: number
  ) {
    if (!data || data.length === 0) {
      return [];
    }

    if (nowStep === 1) {
      return (data as CompanyBookMarkResponse[]).map((item) => ({
        id: item.companyAnalysisId,
        name: item.companyName ?? "",
        industry: item.companyIndustry ?? "",
      }));
    } else if (nowStep === 2) {
      return (data as JobBookMarkResponse[]).map((item) => ({
        id: item.jobRoleAnalysisId,
        name: item.jobRoleName ?? "",
        industry: item.jobRoleAnalysisTitle ?? "",
      }));
    }
  }

  if (!data) return;
  const reports = adaptData(data, nowStep);

  return (
    <>
      <div className="grid  md:grid-cols-3 grid-cols-2 gap-4 ">
        {reports &&
          reports.map((report) => {
            const isSelected =
              nowStep === 1
                ? inputData.companyAnalysisId === report.id
                : inputData.jobRoleAnalysisId === report.id;
            return (
              <div
                onClick={() => handleSelect(report.id)}
                key={report.id}
                className={`${
                  isSelected && selectedStyle
                } ${reportBlockLayout} ${hoverReportBlockLayout}  group active:bg-active`}
              >
                <div className="group-active:bg-active group-active:border-active duration-100 border mr-3 bg-secondary-light rounded-full size-15 flex justify-center items-center font-bold text-primary text-xl aspect-square">
                  {report.name[0]}
                </div>
                <div>
                  <div className="whitespace-normal line-clamp-2 font-semibold text-xl">
                    {report.name}
                  </div>
                  <div className="text-sm line-clamp-2 text-muted-foreground">
                    {report.industry}
                  </div>
                </div>
              </div>
            );
          })}
        <div
          className={`${reportBlockLayout} ${hoverReportBlockLayout} flex-col`}
        >
          <div>{nowStep === 1 ? "기업분석" : "직무분석"} 추가하기</div>
          <div className="text-3xl text-primary">+</div>
        </div>
      </div>
    </>
  );
}

export default ReportList;
