import { useCallback, useEffect, useState } from "react";
import ReportList from "./components/ReportList";
import { useLocation, useNavigate } from "react-router";
import CoverLetterAnalysisLayout from "./components/CoverLetterAnalysisLayout";
import InputQuestion from "./InputQuestion/InputQuestion";
import JobCompanyForm from "./components/JobCompanyForm";

function CoverLetter() {
  const [nowStep, setNowStep] = useState(0);
  const pathname = useLocation().pathname;
  const navigate = useNavigate();

  const handleStep = useCallback((stepNum: number) => {
    setNowStep(stepNum);
    navigate(stepUrl[stepNum]);
  }, []);
  // // 1단계에서 사용할 쿼리
  // const { data: companyData } = useQuery(
  //   ["companies"],
  //   fetchCompanies,
  //   { enabled: nowStep === 1 } // nowStep이 1일 때만 실행
  // );

  // // 2단계에서 사용할 쿼리
  // const { data: jobData } = useQuery(
  //   ["jobs"],
  //   fetchJobs,
  //   { enabled: nowStep === 2 } // nowStep이 2일 때만 실행
  // );

  useEffect(() => {
    switch (pathname) {
      case stepUrl[0]:
        setNowStep(0);
        break;
      case stepUrl[1]:
        setNowStep(1);
        break;
      case stepUrl[2]:
        setNowStep(2);
        break;
      case stepUrl[3]:
        setNowStep(3);
        break;
    }
  }, [pathname]);

  const stepUrl = [
    "/cover-letter",
    "/cover-letter/select-company",
    "/cover-letter/select-job",
    "/cover-letter/input-question",
    `/cover-letter/1`,
  ];

  return (
    <>
      <CoverLetterAnalysisLayout nowStep={nowStep} handleStep={handleStep}>
        {nowStep === 0 && <JobCompanyForm />}
        {(nowStep === 2 || nowStep === 1) && <ReportList nowStep={nowStep} />}
        {nowStep === 3 && <InputQuestion />}
      </CoverLetterAnalysisLayout>
    </>
  );
}

export default CoverLetter;
