package com.ssafy.hellojob.domain.jobroleanalysis.dto.request;

import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleCategory;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class JobRoleAnalysisSearchCondition {

    private String jobRoleName;
    private String jobRoleTitle;
    private JobRoleCategory jobRoleCategory;

}
