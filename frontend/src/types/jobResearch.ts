//특정 기업의 전체 직무 분석 불러오기
export interface getAllJobList {
  jobRoleAnalysisId: number;
  jobRoleName: string;
  jobRoleAnalysisTitle: string;
  jobRoleCategory: string;
  jobRoleViewCount: number;
  jobRoleBookmarkCount: number;
  bookmark: boolean;
  updatedAt: string;
  public: boolean;
}

//상세 직무 정보보
export interface getJobRoleDetail {
  jobRoleAnalysisId: number;
  companyName: string;
  jobRoleName: string;
  jobRoleAnalysisTitle: string;
  jobRoleWork: string;
  jobRoleSkills: string;
  jobRoleRequirements: string;
  jobRolePreferences: string;
  jobRoleEtc: string;
  jobRoleViewCount: number;
  isPublic: boolean;
  jobRoleCategory: string;
  createdAt: string;
  updatedAt: string;
  jobRoleAnalysisBookmarkCount: number;
  bookmark: boolean;
}

//북마크크
export interface postJobBookmarkRequest {
  jobRoleAnalysisId: number;
}

export interface postJobBookmarkResponse {
  jobRoleAnalysisBookmarkId: number;
  jobRoleAnalysisId: number;
}
