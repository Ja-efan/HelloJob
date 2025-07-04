package com.ssafy.hellojob.domain.coverlettercontent.service;

import com.ssafy.hellojob.domain.coverletter.dto.ai.request.CompanyAnalysisDto;
import com.ssafy.hellojob.domain.coverletter.dto.ai.request.ExperienceDto;
import com.ssafy.hellojob.domain.coverletter.dto.ai.request.JobRoleAnalysisDto;
import com.ssafy.hellojob.domain.coverletter.dto.ai.request.ProjectDto;
import com.ssafy.hellojob.domain.coverletter.dto.ai.response.AICoverLetterResponseDto;
import com.ssafy.hellojob.domain.coverletter.dto.request.ContentsDto;
import com.ssafy.hellojob.domain.coverletter.dto.request.CoverLetterSaveRequestDto;
import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import com.ssafy.hellojob.domain.coverletter.repository.CoverLetterRepository;
import com.ssafy.hellojob.domain.coverletter.service.CoverLetterSwotService;
import com.ssafy.hellojob.domain.coverlettercontent.dto.ai.request.AIChatRequestDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.ai.request.AICoverLetterContentDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.request.ChatRequestDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.request.CoverLetterUpdateRequestDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.*;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContentStatus;
import com.ssafy.hellojob.domain.coverlettercontent.repository.CoverLetterContentRepository;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.service.UserReadService;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class CoverLetterContentService {

    private final CoverLetterContentRepository coverLetterContentRepository;
    private final CoverLetterRepository coverLetterRepository;
    private final CoverLetterExperienceService coverLetterExperienceService;
    private final ChatLogService chatLogService;
    private final UserReadService userReadService;
    private final CoverLetterContentReadService coverLetterContentReadService;
    private final CoverLetterSwotService coverLetterSwotService;

    @Transactional
    public List<CoverLetterContent> createContents(User user, CoverLetter coverLetter, List<ContentsDto> contentsDto) {
        List<CoverLetterContent> contents = new ArrayList<>();

        for (ContentsDto content : contentsDto) {

            CoverLetterContent newCoverLetterContent = CoverLetterContent.builder()
                    .coverLetter(coverLetter)
                    .contentStatus(CoverLetterContentStatus.PENDING)
                    .contentQuestion(content.getContentQuestion())
                    .contentNumber(content.getContentNumber())
                    .contentLength(content.getContentLength())
                    .contentFirstPrompt(content.getContentFirstPrompt())
                    .build();

            log.debug("자기소개서 입력 경험 아이디: {} 프로젝트 아이디: {}", content.getContentExperienceIds(), content.getContentProjectIds());

            coverLetterContentRepository.save(newCoverLetterContent);
            coverLetterExperienceService.saveCoverLetterExperience(content.getContentExperienceIds(), user, newCoverLetterContent);
            coverLetterExperienceService.saveCoverLetterProject(content.getContentProjectIds(), user, newCoverLetterContent);

            contents.add(newCoverLetterContent);
        }
        return contents;
    }

    // ai 자기소개서 초안 응답 저장
    @Transactional
    public void appendDetail(List<CoverLetterContent> contents, List<AICoverLetterResponseDto> aiResponses) {
        Map<Integer, String> aiMap = aiResponses.stream()
                .collect(Collectors.toMap(AICoverLetterResponseDto::getContent_number, AICoverLetterResponseDto::getCover_letter));

        for (CoverLetterContent content : contents) {
            String contentDetail = aiMap.get(content.getContentNumber());
            content.updateContentDetail(contentDetail);
        }
    }

    // 자기소개서 문항별 조회 응답
    @Transactional
    public CoverLetterContentDto getCoverLetterContent(Integer userId, Integer contentId) {

        userReadService.findUserByIdOrElseThrow(userId);
        CoverLetterContent coverLetterContent = coverLetterContentReadService.findCoverLetterContentByIdOrElseThrow(contentId);
        coverLetterContentReadService.checkCoverLetterContentValidation(userId, coverLetterContent);

        List<Integer> contentExperienceIds =
                coverLetterExperienceService.getCoverLetterExperienceIds(contentId);

        List<Integer> contentProjectIds =
                coverLetterExperienceService.getCoverLetterProjectIds(contentId);

        List<ChatMessageDto> contentChatLog = chatLogService.getContentChatLog(coverLetterContent.getContentId());

        return CoverLetterContentDto.builder()
                .contentId(coverLetterContent.getContentId())
                .contentQuestion(coverLetterContent.getContentQuestion())
                .contentNumber(coverLetterContent.getContentNumber())
                .contentLength(coverLetterContent.getContentLength())
                .contentDetail(coverLetterContent.getContentDetail())
                .contentFirstPrompt(coverLetterContent.getContentFirstPrompt())
                .contentExperienceIds(contentExperienceIds)
                .contentProjectIds(contentProjectIds)
                .contentUpdatedAt(coverLetterContent.getUpdatedAt())
                .contentChatLog(contentChatLog)
                .build();
    }

    public List<ContentQuestionStatusDto> getCoverLetterContentQuestionStatues(Integer coverLetterId) {
        return coverLetterContentRepository.getCoverLetterContentStatuses(coverLetterId);
    }

    public boolean isWholeContentCompleted(List<ContentQuestionStatusDto> statuses) {
        boolean result = true;
        for (ContentQuestionStatusDto status : statuses) {
            if (status.getContentStatus() != CoverLetterContentStatus.COMPLETED) {
                result = false;
                break;
            }
        }
        return result;
    }

    public void saveAllContents(CoverLetter coverLetter, List<CoverLetterSaveRequestDto> requestDto) {
        List<CoverLetterContent> contents = coverLetterContentRepository.findByCoverLetter(coverLetter);

        Set<Integer> validContentId = contents.stream()
                .map(CoverLetterContent::getContentId)
                .collect(Collectors.toSet());

        // 자기소개서에 있는 contentId 인지 검사
        for (CoverLetterSaveRequestDto dto : requestDto) {
            if (!validContentId.contains(dto.getContentId())) {
                throw new BaseException(ErrorCode.COVER_LETTER_CONTENT_NOT_MATCHED_COVER_LETTER);
            }
        }

        // 넘어온 애들만 detail 업데이트
        for (CoverLetterContent content : contents) {
            requestDto.stream()
                    .filter(dto -> dto.getContentId().equals(content.getContentId()))
                    .findFirst()
                    .ifPresent(dto -> {
                        content.updateContentDetail(dto.getContentDetail());
                    });
            content.updateContentStatus(CoverLetterContentStatus.COMPLETED);
        }
    }

    // 자기소개서 id에 해당하는 contentId 리스트 반환
    public List<Integer> getContentIdsByCoverLetterId(Integer coverLetterId) {
        return coverLetterContentRepository
                .findContentIdByCoverLetterId(coverLetterId);
    }

    @Transactional
    public Map<String, String> updateCoverLetterContent(Integer userId, Integer contentId, CoverLetterUpdateRequestDto requestDto) {

        userReadService.findUserByIdOrElseThrow(userId);
        CoverLetterContent content = coverLetterContentReadService.findCoverLetterContentByIdOrElseThrow(contentId);
        coverLetterContentReadService.checkCoverLetterContentValidation(userId, content);

        if (requestDto.getContentStatus() == CoverLetterContentStatus.PENDING) {
            throw new BaseException(ErrorCode.COVER_LETTER_CONTENT_ALREADY_START);
        }

        log.debug("🌞 저장 요청된 content 내용: {}, status: {}", requestDto.getContentDetail(), requestDto.getContentStatus());
        content.updateCoverLetterContent(requestDto);
        Integer coverLetterId = content.getCoverLetter().getCoverLetterId();

        log.debug("👉 저장된 content 내용: {}, status: {}", content.getContentDetail(), content.getContentStatus());

        List<ContentQuestionStatusDto> statuses = getCoverLetterContentQuestionStatues(coverLetterId);
        boolean isWholeContentCompleted = isWholeContentCompleted(statuses);

        // 전체 완료인 경우 자기소개서 finish 처리
        if (isWholeContentCompleted) {
            content.getCoverLetter().updateFinish(true);
        } else { // 아닌 경우 updatedAt만 반영
            content.getCoverLetter().updateFinish(false);
            coverLetterRepository.touch(coverLetterId);
        }

        if (content.getContentStatus() == CoverLetterContentStatus.IN_PROGRESS)
            return Map.of("message", "자기소개서가 임시 저장되었습니다.");

        return Map.of("message", "자기소개서가 저장되었습니다.");
    }

    public List<CoverLetterOnlyContentDto> getWholeContentDetail(Integer coverLetterId) {
        return coverLetterContentRepository.findContentByCoverLetterId(coverLetterId);
    }

    public ChatResponseDto sendChat(Integer userId, Integer contentId, ChatRequestDto requestDto) {
        AIChatRequestDto aiChatRequestDto = getAIChatRequestDto(userId, contentId, requestDto);
        CoverLetterContent content = coverLetterContentReadService.findCoverLetterContentByIdOrElseThrow(contentId);
        return chatLogService.sendChat(content, aiChatRequestDto);
    }

    public AIChatRequestDto getAIChatRequestDto(Integer userId, Integer contentId, ChatRequestDto requestDto) {

        userReadService.findUserByIdOrElseThrow(userId);
        CoverLetterContent content = coverLetterContentReadService.findCoverLetterContentByIdOrElseThrow(contentId);
        coverLetterContentReadService.checkCoverLetterContentValidation(userId, content);

        Integer coverLetterId = coverLetterContentRepository.findCoverLetterIdByContentId(contentId)
                .orElseThrow(() -> new BaseException(ErrorCode.COVER_LETTER_NOT_FOUND));

        CoverLetter coverLetter = coverLetterRepository.findFullCoverLetterDetail(coverLetterId);

        List<ChatMessageDto> chatLog = chatLogService.getContentChatLog(contentId);
        log.debug("🌞 chatLog size : {}", chatLog.size());
        List<ChatMessageDto> chatRecentHistory = chatLog.subList(Math.max(chatLog.size() - 10, 0), chatLog.size());

        return AIChatRequestDto.builder()
                .user_message(requestDto.getUserMessage())
                .chat_history(chatRecentHistory)
                .company_analysis(CompanyAnalysisDto.from(coverLetter.getCompanyAnalysis(), coverLetterSwotService.getSWOTDto(coverLetter)))
                .job_role_analysis(
                        coverLetter.getJobRoleSnapshot() != null
                                ? JobRoleAnalysisDto.from(coverLetter.getJobRoleSnapshot())
                                : null)
                .experiences(content.getExperiences().stream()
                        .map(cle -> cle.getExperience())
                        .filter(Objects::nonNull)
                        .map(ExperienceDto::from)
                        .toList()
                )
                .projects(content.getExperiences().stream()
                        .map(cle -> cle.getProject())
                        .filter(Objects::nonNull)
                        .map(ProjectDto::from)
                        .toList()
                )
                .cover_letter(AICoverLetterContentDto.from(content))
                .build();
    }
}
