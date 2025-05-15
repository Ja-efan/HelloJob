package com.ssafy.hellojob.domain.coverlettercontent.entity;

import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import com.ssafy.hellojob.global.common.domain.BaseTimeEntity;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.ColumnDefault;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "chat_log")
public class ChatLog extends BaseTimeEntity {

    @Id
    private Integer coverLetterContentId;

    @OneToOne
    @MapsId
    @JoinColumn(name = "cover_letter_content_id")
    private CoverLetterContent coverLetterContent;

    @Column(name = "chat_log_content", columnDefinition = "TEXT", nullable = false)
    private String chatLogContent;

    @Column(name = "updated_count", nullable = false)
    @ColumnDefault("0")
    private Integer updatedCount = 0;

    @Builder
    public ChatLog(CoverLetterContent coverLetterContent, String chatLogContent, Integer updatedCount) {
        this.coverLetterContent = coverLetterContent;
        this.chatLogContent = chatLogContent;
        this.updatedCount = updatedCount;
    }

    public void updateChatLog(String chatLogContent) {
        this.chatLogContent = chatLogContent;
        this.updatedCount += 1;
    }
}
