package com.ssafy.hellojob.domain.jobrolesnapshot.controller;

import com.ssafy.hellojob.domain.jobrolesnapshot.dto.response.JobRoleSnapshotResponseDto;
import com.ssafy.hellojob.domain.jobrolesnapshot.service.JobRoleSnapshotService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
@RequestMapping("api/v1/job-role-snapshot")
public class JobRoleSnapshotController {

    private final JobRoleSnapshotService jobRoleSnapshotService;

    @GetMapping("/{jobRoleSnapshotId}")
    public JobRoleSnapshotResponseDto getJobRoleSnapshot(@PathVariable Integer jobRoleSnapshotId) {
        return jobRoleSnapshotService.getJobRoleSnapshot(jobRoleSnapshotId);
    }
}
