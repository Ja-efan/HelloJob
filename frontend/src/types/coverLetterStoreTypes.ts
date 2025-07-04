import { CoverLetterRequestContent } from "./coverLetterTypes";

export type ChatMessage = {
  sender: "user" | "ai";
  message: string;
};

export type ChatStore = {
  chatLog: ChatMessage[];
  setChatLog: (chatLog: ChatMessage[]) => void;
  addUserMessage: (message: string) => void;
  addAiMessage: (message: string) => void;
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
  resetSelectCompany: () => void;
}

export interface CoverLetterInputStoreType {
  inputData: {
    companyAnalysisId: number | null;
    jobRoleAnalysisId: number | null;
    coverLetterTitle: string;
    contents: CoverLetterRequestContent[];
  };

  setCoverLetterTitle: (title: string) => void;
  setCompanyAnalysisId: (id: number | null) => void;
  setJobRoleAnalysisId: (id: number | null) => void;
  setContentProjectIds: (contentIndex: number, projectIds: number[]) => void;
  setContentExperienceIds: (
    contentIndex: number,
    experienceIds: number[]
  ) => void;

  addQuestion: () => void;
  setAllQuestions: (contents: CoverLetterRequestContent[]) => void;
  updateContent: (
    contentIndex: number,
    updatedData: Partial<CoverLetterRequestContent>
  ) => void;
  resetAllInputs: () => void;
}
