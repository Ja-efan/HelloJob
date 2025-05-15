import { companyAnalysisApi, getCompanies } from "@/api/companyApi";
import { CompanyBookMarkResponse } from "@/types/coverLetterTypes";
import { useQuery } from "@tanstack/react-query";

interface GetCompaniesResponse {
  companyName: string;
  companyIndustry: string;
  companyLocation: string;
  companySize: string;
  id: number;
  updatedAt: string;
}

export const useGetCompanies = (companyName: string) => {
  return useQuery<GetCompaniesResponse[], Error>({
    queryKey: ["company-name", companyName],
    queryFn: async () => {
      const response = await getCompanies(companyName);
      return response.data;
    },
    enabled: companyName.trim().length > 0,
  });
};

export const useGetCompanyBookMarks = (companyId: number) => {
  return useQuery<CompanyBookMarkResponse[]>({
    queryKey: ["companyBookMark", companyId],
    queryFn: async () => {
      const response = await companyAnalysisApi.getCompanyBookMarks(companyId);
      console.log(response.data);
      return response.data;
    },
  });
};
