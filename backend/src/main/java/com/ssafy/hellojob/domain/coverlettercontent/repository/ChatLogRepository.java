package com.ssafy.hellojob.domain.coverlettercontent.repository;

import com.ssafy.hellojob.domain.coverlettercontent.entity.ChatLog;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface ChatLogRepository extends JpaRepository<ChatLog, Integer> {

    @Query("SELECT c.chatLogContent FROM ChatLog c WHERE c.coverLetterContent.contentId = :contentId")
    String findChatLogContentById(@Param("contentId") Integer contentId);
}
