import { postBookmarkRequest } from "@/types/coporateResearch";
import { authApi } from "./instance";

export const corporateListApi = {
  getCorporateList: (companyName: string) => {
    return authApi.get(`/api/v1/company/search?companyName=${companyName}`);
  },
};

export const corporateReportApi = {
  getCorporateReportList: (companyId: number) => {
    return authApi.get(`/api/v1/company-analysis/search/${companyId}`);
  },
  getCorporateReportDetail: (companyAnalysisId: number) => {
    return authApi.get(`/api/v1/company-analysis/${companyAnalysisId}`);
  },
  postBookmark: (postBookmarkRequest: postBookmarkRequest) => {
    return authApi.post(`/api/v1/company-analysis/bookmark`, {
      postBookmarkRequest,
    });
  },
  deleteBookmark: (companyAnalysisId: number) => {
    return authApi.delete(
      `/api/v1/company-analysis/bookmark/${companyAnalysisId}`
    );
  },
};
