package com.ssafy.hellojob.domain.user.entity;

import com.ssafy.hellojob.global.common.domain.BaseTimeEntity;
import jakarta.persistence.*;
import lombok.*;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@Table(name = "user")
public class User extends BaseTimeEntity {

    @Id
    @Column(name = "user_id")
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer userId;

    @Column(name = "email", length = 100, nullable = false)
    private String email;

    @Column(name = "nickname", length = 30)
    private String nickname;

    @Enumerated(EnumType.STRING)
    @Column(name = "provider")
    private Provider provider;

    @Column(name = "provider_id", nullable = false)
    private String providerId;

    @Column(name = "withdraw", nullable = false)
    private Boolean withdraw;

    @Column(name = "token", nullable = false)
    private Integer token;

    @Builder
    public User(String email, String nickname, Provider provider, String providerId) {
        this.email = email;
        this.nickname = nickname;
        this.provider = provider;
        this.providerId = providerId;
        this.withdraw = false;
        this.token = 3;
    }

    public void decreaseToken() {
        if (this.token > 0) {
            this.token -= 1;
        }
    }

    public void changeNickname(String nickname) {
        this.nickname = nickname;
    }

}
