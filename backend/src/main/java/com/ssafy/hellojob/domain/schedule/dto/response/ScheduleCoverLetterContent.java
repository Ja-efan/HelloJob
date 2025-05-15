package com.ssafy.hellojob.domain.schedule.dto.response;

import lombok.*;

@Getter
@Builder
@AllArgsConstructor
public class ScheduleCoverLetterContent {

    private Integer contentId;
    private Integer contentNumber;
    private String contentQuestion;
    private Integer contentLength;
    private String contentDetail;

}
