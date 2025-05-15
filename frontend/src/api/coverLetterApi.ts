import { CoverLetterPostRequest } from "@/types/coverLetterTypes";
import { authApi } from "./instance";

export const coverLetterApi = {
  postCoverLetter: (postCoverLetterRequest: CoverLetterPostRequest) => {
    return authApi.post("/api/v1/cover-letter", { postCoverLetterRequest });
  },
  getCoverLetter: (coverLetterId: number, coverLetterNumber: number) => {
    return authApi.get(
      `/api/v1/cover-letter/${coverLetterId}/${coverLetterNumber}`
    );
  },

  draftCoverLetter: (coverLetterId: number, coverLetterNumber: number) => {
    return authApi.patch(
      `/api/v1/cover-letter/${coverLetterId}/${coverLetterNumber}`
    );
  },
  saveCoverLetter: (coverLetterId: number) => {
    return authApi.patch(`/api/v1/cover-letter/${coverLetterId}`);
  },
  sendMessage: (coverLetterId: number, coverLetterNumber: number) => {
    return authApi.post(
      `/api/v1/cover-letter/${coverLetterId}/${coverLetterNumber}/chat`
    );
  },
  deleteCoverLetter: (coverLetterId: number) => {
    return authApi.delete(`/api/v1/cover-letter/${coverLetterId}`);
  },
  getConnectAnalysisInfo: (
    CoverLetterId: number,
    jobRoleSnapshotId: number
  ) => {
    return authApi.get(
      `/api/v1/cover-letter/${CoverLetterId}/${jobRoleSnapshotId}`
    );
  },
};
