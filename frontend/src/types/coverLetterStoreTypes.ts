import { CoverLetterRequestContent } from "./coverLetterTypes";

export type ChatMessage = {
  sender: "USER" | "AI";
  message: string;
};

export type ChatStore = {
  chatLog: ChatMessage[];
  addUserMessage: (message: string) => void;
};

export interface CompanyState {
  companyId: number;
  companyName: string;
  companySize: string;
  companyLocation: string;
}

export interface SelectCompanyState {
  company: CompanyState;

  setSelectCompany: (company: CompanyState) => void;
}

export interface CoverLetterInputStoreType {
  inputData: {
    companyAnalysisId: number | null;
    jobRoleAnalysisId: number | null;
    contents: CoverLetterRequestContent[];
  };

  setCompanyAnalysisId: (id: number | null) => void;
  setJobRoleAnalysisId: (id: number | null) => void;
  addQuestion: () => void;
}
