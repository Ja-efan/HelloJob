import { PostProjectRequest } from "@/api/experienceApi";
import { projectApi } from "@/api/projectApi";
import { useMutation, useQuery } from "@tanstack/react-query";

// (projectFormData: PostProjectRequest)

export const useGetProjects = () => {
  return useQuery({
    queryKey: ["projects"],
    queryFn: async () => {
      const response = await projectApi.getProjects();
      console.log(response);
      return response.data;
    },
  });
};

export const usePostProject = () => {
  return useMutation({
    mutationKey: ["postProject"],
    mutationFn: async (projectFormData: PostProjectRequest) => {
      const response = await projectApi.postProject(projectFormData);
      return response.data;
    },
  });
};
