import {
  ChatStore,
  CoverLetterInputStoreType,
} from "@/types/coverLetterStoreTypes";
import { toast } from "sonner";
import { create } from "zustand";

export const useCoverLetterStore = create<ChatStore>((set) => ({
  chatLog: [
    {
      sender: "AI",
      message:
        "안녕하세요! 초안작성을 도와드립니다! 궁금하신점을 수정할 수 있도록 도와드려요!",
    },
  ],
  addUserMessage: (message: string) => {
    set((prev) => ({
      chatLog: [...prev.chatLog, { sender: "USER", message }],
    }));
  },
  addAiMessage: (message: string) => {
    set((prev) => ({
      chatLog: [...prev.chatLog, { sender: "AI", message }],
    }));
  },
}));

export const useCoverLetterInputStore = create<CoverLetterInputStoreType>(
  (set) => ({
    inputData: {
      companyAnalysisId: null,
      jobRoleAnalysisId: null,
      contents: [
        {
          contentQuestion: "",
          contentNumber: 1,
          contentExperienceIds: [],
          contentProjectIds: [],
          contentLength: 0,
          contentFirstPrompt: "",
        },
      ],
    },

    setCompanyAnalysisId: (id: number | null) => {
      set((state) => ({
        inputData: {
          ...state.inputData,
          companyAnalysisId: id,
        },
      }));
    },

    setJobRoleAnalysisId: (id: number | null) => {
      set((state) => ({
        inputData: {
          ...state.inputData,
          jobRoleAnalysisId: id,
        },
      }));
    },

    addQuestion: () =>
      set((state) => {
        if (state.inputData.contents.length >= 10) {
          toast.warning("최대 10개의 문항만 추가할 수 있습니다.");
          return state;
        }

        return {
          inputData: {
            ...state.inputData,
            contents: [
              ...state.inputData.contents,
              {
                contentQuestion: "",
                contentNumber: state.inputData.contents.length + 1,
                contentExperienceIds: [],
                contentProjectIds: [],
                contentLength: 0,
                contentFirstPrompt: "",
              },
            ],
          },
        };
      }),
  })
);
