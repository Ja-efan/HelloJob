export type ContentQuestionStatusType = "COMPLETED" | "IN_PROGRESS" | "PENDING";
export type SenderType = "AI" | "USER";

export interface QuestionStatus {
  contentNumber: number;
  contentStatus: ContentQuestionStatusType;
}

export interface QuestionStepProps {
  QuestionStatuses: QuestionStatus[];
  handleSelectQuestion: (selectNum: number) => void;
  selectQuestion: number;
}

export interface QuestionStatus {
  contentNumber: number;
  contentStatus: ContentQuestionStatusType;
}

export interface ContentChatLogType {
  sender: SenderType;
  message: string;
}

export interface CoverLetterContentType {
  contentQuestion: string;
  contentNumber: number;
  contentLength: number;
  contentDetail: string;
  contentExperienceIds: number[];
  contentProjectIds: number[];
  contentFirstPrompt: string;
  contentStatus: ContentQuestionStatusType;
  contentChatLog: ContentChatLogType[];
  contentUpdatedAt: string;
}
export interface CoverLetterSummaryType {
  totalContentQuestionCount: number;
  contentQuestionStatuses: QuestionStatus[];
  companyAnalysisId: number;
  jobRoleSnapshotId: number;
  coverLetterUpdatedAt: string;
}

export interface CoverLetterResponse {
  coverLetterId: number;
  summary: CoverLetterSummaryType;
  content: CoverLetterContentType;
}

//Post Request Type
export interface CoverLetterPostRequest {
  companyAnalysisId: number;
  jobRoleAnalysisId: number | null;
  contents: CoverLetterRequestContent[];
}

export interface CoverLetterRequestContent {
  contentQuestion: string;
  contentNumber: number;
  contentExperienceIds: number[];
  contentProjectIds: number[];
  contentLength: number;
  contentFirstPrompt: string;
}

export interface ReportListProps {
  nowStep: number;
}

// bookMark 타입
export interface JobBookMarkResponse {
  jobRoleAnalysisId: number;
  companyName: string;
  jobRoleName: string;
  jobRoleAnalysisTitle: string;
  jobRoleCategory: string;
  jobRoleViewCount: number;
  jobRoleBookmarkCount: number;
  bookmark: boolean;
  updatedAt: string;
  public: boolean;
}

export interface CompanyBookMarkResponse {
  companyAnalysisBookmarkId: number;
  companyAnalysisId: number;
  companyName: string;
  createdAt: string;
  companyViewCount: number;
  companyLocation: string;
  companySize: string;
  companyIndustry: string;
  companyAnalysisBookmarkCount: number;
  bookmark: boolean;
  public: boolean;
}
