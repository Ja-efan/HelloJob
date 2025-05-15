import { authApi } from "./instance";

export const getCompanies = (companyName: string) => {
  return authApi.get("/api/v1/company/search", {
    params: { companyName },
  });
};

export const companyAnalysisApi = {
  getCompanyAnalyses: () => {
    return authApi.get("/api/v1/company-analysis/all-analysis");
  },

  getCompanyBookMarks: (companyId: number) => {
    return authApi.get("/api/v1/company-analysis/bookmark", {
      params: { companyId },
    });
  },
};
