import { coverLetterApi } from "@/api/coverLetterApi";
import { useQuery } from "@tanstack/react-query";

export const useGetCoverLetter = (
  coverLetterId: number,
  coverLetterNumber: number
) => {
  return useQuery({
    queryKey: ["cover-letter-number", "cover-letter"],
    queryFn: async () => {
      const response = await coverLetterApi.getCoverLetter(
        coverLetterId,
        coverLetterNumber
      );
      return response;
    },
  });
};
