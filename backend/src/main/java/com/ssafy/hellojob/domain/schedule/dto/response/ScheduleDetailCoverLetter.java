package com.ssafy.hellojob.domain.schedule.dto.response;

import lombok.*;

import java.util.List;

@Getter
@Builder
@AllArgsConstructor
public class ScheduleDetailCoverLetter {

    private String coverLetterTitle;
    private Integer coverLetterId;
    private boolean finish;
    private List<ScheduleCoverLetterContent> scheduleCoverLetterContents;

}
