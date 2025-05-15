import { postJobBookmarkRequest } from "@/types/jobResearch";
import { api } from "./api";
import { authApi } from "./instance";

export const jobRoleAnalysis = {
  getJobDetail: (jobRoleAnalysisId: number) => {
    return authApi.get(`/api/v1/job_role_analysis/${jobRoleAnalysisId}`);
  },

  getAllJobList: (companyId: number) => {
    return authApi.get(`/api/v1/job-role-analysis/${companyId}/search`);
  },

  getJobBookMarks: (companyId: number) => {
    return api.get(`/api/v1/job-role-analysis/bookmark`, {
      params: { companyId },
    });
  },

  postBookmark: (postJobBookmarkRequest: postJobBookmarkRequest) => {
    return authApi.post(`/api/v1/job-role-analysis/bookmark`, {
      postJobBookmarkRequest,
    });
  },
  deleteBookmark: (jobRoleAnalysisBookmarkId: number) => {
    return authApi.delete(
      `/api/v1/job-role-analysis/bookmark/${jobRoleAnalysisBookmarkId}`
    );
  },
};
