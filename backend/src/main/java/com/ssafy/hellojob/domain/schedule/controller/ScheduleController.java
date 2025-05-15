package com.ssafy.hellojob.domain.schedule.controller;

import com.ssafy.hellojob.domain.schedule.dto.request.ScheduleAddRequestDto;
import com.ssafy.hellojob.domain.schedule.dto.request.ScheduleUpdateScheduleCoverLetterRequestDto;
import com.ssafy.hellojob.domain.schedule.dto.request.ScheduleUpdateScheduleStatusRequestDto;
import com.ssafy.hellojob.domain.schedule.dto.response.ScheduleDetailResponseDto;
import com.ssafy.hellojob.domain.schedule.dto.response.ScheduleIdResponseDto;
import com.ssafy.hellojob.domain.schedule.dto.response.ScheduleListResponseDto;
import com.ssafy.hellojob.domain.schedule.service.ScheduleService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/schedule")
public class ScheduleController {

    private final ScheduleService scheduleService;

    // 일정 추가
    @PostMapping()
    public ScheduleIdResponseDto ScheduleAdd(@Valid @RequestBody ScheduleAddRequestDto requestDto,
                                              @AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();

        ScheduleIdResponseDto responseDto = scheduleService.addSchedule(requestDto, userId);

        return responseDto;

    }

    // 일정 삭제
    @DeleteMapping("{scheduleId}")
    public void ScheduleDelete(@PathVariable("scheduleId") Integer scheduleId,
                               @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();

        scheduleService.deleteSchedule(scheduleId, userId);

    }

    // 일정 상태 수정
    @PatchMapping("/{scheduleId}/status")
    public ScheduleIdResponseDto ScheduleUpdateStatus(@Valid @RequestBody ScheduleUpdateScheduleStatusRequestDto requestDto,
                                                      @PathVariable("scheduleId") Integer scheduleId,
                                                      @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();

        ScheduleIdResponseDto responseDto = scheduleService.updateScheduleStatus(requestDto, scheduleId, userId);
        return responseDto;
    }

    // 일정 자기소개서 수정
    @PatchMapping("/{scheduleId}/cover-letter")
    public ScheduleIdResponseDto ScheduleUpdateCoverLetter(@Valid @RequestBody ScheduleUpdateScheduleCoverLetterRequestDto requestDto,
                                                            @PathVariable("scheduleId") Integer scheduleId,
                                                            @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();

        ScheduleIdResponseDto responseDto = scheduleService.updateScheduleCoverLetter(requestDto, scheduleId, userId);
        return responseDto;
    }

    // 일정 전체 수정
    @PutMapping("/{scheduleId}")
    public ScheduleIdResponseDto ScheduleUpdate(@Valid @RequestBody ScheduleAddRequestDto requestDto,
                                                @PathVariable("scheduleId") Integer scheduleId,
                                                @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();
        ScheduleIdResponseDto responseDto = scheduleService.updateSchedule(requestDto, scheduleId, userId);
        return responseDto;
    }

    // 일정 전체 조회
    @GetMapping()
    public List<ScheduleListResponseDto> ScheduleList(@AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();
        List<ScheduleListResponseDto> responseDto = scheduleService.allSchedule(userId);

        return responseDto;
    }

    // 일정 상세 조회
    @GetMapping("/{scheduleId}")
    public ScheduleDetailResponseDto ScheduleDetail(@PathVariable("scheduleId") Integer scheduleId,
                                                    @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();
        ScheduleDetailResponseDto responseDto = scheduleService.detailSchedule(scheduleId, userId);

        return responseDto;
    }


}
