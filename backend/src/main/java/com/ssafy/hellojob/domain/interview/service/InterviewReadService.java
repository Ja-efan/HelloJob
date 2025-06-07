package com.ssafy.hellojob.domain.interview.service;

import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import com.ssafy.hellojob.domain.interview.entity.*;
import com.ssafy.hellojob.domain.interview.repository.*;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.exception.BaseException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import static com.ssafy.hellojob.global.exception.ErrorCode.*;

@Service
@RequiredArgsConstructor
public class InterviewReadService {

    private final CoverLetterInterviewRepository coverLetterInterviewRepository;
    private final CsQuestionBankRepository csQuestionBankRepository;
    private final PersonalityQuestionBankRepository personalityQuestionBankRepository;
    private final CoverLetterQuestionBankRepository coverLetterQuestionBankRepository;
    private final InterviewQuestionMemoRepository interviewQuestionMemoRepository;
    private final InterviewAnswerRepository interviewAnswerRepository;
    private final InterviewVideoRepository interviewVideoRepository;
    private final InterviewRepository interviewRepository;

    public InterviewVideo findInterviewVideoByIdWithInterviewAndCoverLetterInterviewOrElseThrow(Integer interviewVideoId) {
        return interviewVideoRepository.findByIdWithInterviewAndCoverLetterInterview(interviewVideoId)
                .orElseThrow(() -> new BaseException(INTERVIEW_VIDEO_NOT_FOUND));
    }

    public CoverLetterInterview findCoverLetterInterviewByUserAndCoverLetterOrElseThrow(User user, CoverLetter coverLetter) {
        return coverLetterInterviewRepository.findByUserAndCoverLetter(user, coverLetter)
                .orElseThrow(() -> new BaseException(COVER_LETTER_QUESTION_MISMATCH));
    }

    public Interview findInterviewByIdAndUserOrElseThrow(Integer interviewId, User user) {
        return interviewRepository.findByUserAndInterviewId(user, interviewId)
                .orElseThrow(() -> new BaseException(INTERVIEW_MISMATCH));
    }

    public CsQuestionBank findCsQuestionByIdOrElseThrow(Integer csQuestionBankId) {
        return csQuestionBankRepository.findById(csQuestionBankId)
                .orElseThrow(() -> new BaseException(CS_QUESTION_NOT_FOUND));
    }

    public PersonalityQuestionBank findPersonalityQuestionByIdOrElseThrow(Integer personalityQuestionBankId) {
        return personalityQuestionBankRepository.findById(personalityQuestionBankId)
                .orElseThrow(() -> new BaseException(PERSONALITY_QUESTION_NOT_FOUND));
    }

    public CoverLetterQuestionBank findCoverLetterQuestionByIdWithCoverLetterOrElseThrow(Integer coverLetterQuestionBankId) {
        return coverLetterQuestionBankRepository.findByIdWithCoverLetterInterview(coverLetterQuestionBankId)
                .orElseThrow(() -> new BaseException(COVER_LETTER_QUESTION_NOT_FOUND));
    }

    public InterviewQuestionMemo findInterviewQuestionMemoWithUserByIdOrElseThrow(Integer interviewQuestionMemoId) {
        return interviewQuestionMemoRepository.findByIdWithUser(interviewQuestionMemoId)
                .orElseThrow(() -> new BaseException(INTERVIEW_QUESTION_MEMO_NOT_FOUND));
    }

    public InterviewQuestionMemo findInterviewQuestionMemoByUserAndCsQuestionOrElseReturnNull(User user, CsQuestionBank csQuestionBank) {
        return interviewQuestionMemoRepository.findByUserAndCsQuestionBank(user, csQuestionBank)
                .orElse(null);
    }

    public InterviewQuestionMemo findInterviewQuestionMemoByUserAndPersonalityQuestionOrElseReturnNull(User user, PersonalityQuestionBank personalityQuestionBank) {
        return interviewQuestionMemoRepository.findByUserAndPersonalityQuestionBank(user, personalityQuestionBank)
                .orElse(null);
    }

    public InterviewQuestionMemo findInterviewQuestionMemoByUserAndCoverLetterQuestionOrElseReturnNull(User user, CoverLetterQuestionBank coverLetterQuestionBank) {
        return interviewQuestionMemoRepository.findByUserAndCoverLetterQuestionBank(user, coverLetterQuestionBank)
                .orElse(null);
    }

    public CoverLetterInterview findCoverLetterInterviewByIWithUserdOrElseThrow(Integer coverLetterInterviewId) {
        return coverLetterInterviewRepository.findByIdWithUser(coverLetterInterviewId)
                .orElseThrow(() -> new BaseException(COVER_LETTER_INTERVIEW_NOT_FOUND));
    }

    public InterviewVideo findInterviewVideoByIdOrElseThrow(Integer interveiwVideoId){
        return interviewVideoRepository.findById(interveiwVideoId)
                .orElseThrow(() -> new BaseException(INTERVIEW_VIDEO_NOT_FOUND));
    }

    public InterviewAnswer findInterviewAnswerByIdOrElseThrow(Integer interviewAnswerId){
        return interviewAnswerRepository.findById(interviewAnswerId)
                .orElseThrow(() -> new BaseException(INTERVIEW_ANSWER_NOT_FOUND));
    }

    public Interview findInterviewById(Integer interviewId){
        return interviewRepository.findById(interviewId)
                .orElseThrow(() -> new BaseException(INTERVIEW_NOT_FOUND));
    }

    public CoverLetterInterview findCoverLetterInterviewById(Integer interviewId){
        return coverLetterInterviewRepository.findById(interviewId)
                .orElseThrow(() -> new BaseException(INTERVIEW_NOT_FOUND));
    }

    public CoverLetterInterview findCoverLetterInterviewByCoverLetter(CoverLetter coverLetter){
        return coverLetterInterviewRepository.findByCoverLetter(coverLetter)
                .orElseThrow(() -> new BaseException(INTERVIEW_NOT_FOUND));
    }

    public Integer countTotalQuestions(Integer videoId){
        return interviewVideoRepository.countTotalAnswer(videoId);
    }

    public Integer countSavedAnswers(Integer videoId){
        return interviewVideoRepository.countSavedAnswer(videoId);
    };

}
