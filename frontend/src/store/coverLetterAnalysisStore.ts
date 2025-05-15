import {
  CompanyState,
  SelectCompanyState,
} from "@/types/coverLetterStoreTypes";
import { create } from "zustand/react";

export const useSelectCompanyStore = create<SelectCompanyState>((set) => ({
  company: {
    companyId: 0,
    companyName: "",
    companyLocation: "",
    companySize: "",
  },

  setSelectCompany: (companyData: CompanyState) => {
    set({
      company: {
        companyId: companyData.companyId,
        companyName: companyData.companyName,
        companySize: companyData.companySize,
        companyLocation: companyData.companyLocation,
      },
    });
  },
}));

export type JobRole =
  | "서버백엔드개발자"
  | "프론트엔드개발자"
  | "안드로이드개발자"
  | "iOS개발자"
  | "크로스플랫폼앱개발자"
  | "게임클라이언트개발자"
  | "게임서버개발자"
  | "DBA"
  | "빅데이터엔지니어"
  | "인공지능머신러닝"
  | "devops시스템엔지니어"
  | "정보보안침해대응"
  | "QA엔지니어"
  | "개발PM"
  | "HW펌웨어개발"
  | "SW솔루션"
  | "헬스테크"
  | "VRAR3D"
  | "블록체인"
  | "기술지원"
  | "기타"
  | string;
export interface CompanyJobState {
  jobRoleCategory: JobRole;
  setJobRoleCategory: (category: JobRole) => void;
}

export const useSelectJobStore = create<CompanyJobState>((set) => ({
  jobRoleCategory: "",

  setJobRoleCategory: (category: JobRole) => {
    set({ jobRoleCategory: category });
  },
}));
