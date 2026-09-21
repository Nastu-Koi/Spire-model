using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Entities.CardRewardAlternatives;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Merchant;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Entities.RestSite;
using MegaCrit.Sts2.Core.Events;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.GameActions;
using MegaCrit.Sts2.Core.Hooks;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Models.Cards;
using MegaCrit.Sts2.Core.Models.Events;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Rewards;
using MegaCrit.Sts2.Core.Rooms;
using MegaCrit.Sts2.Core.Runs;
using MegaCrit.Sts2.Core.Saves.Runs;

namespace RunRecorder;

internal sealed class Snapshot
{
	private sealed record Identity(long Id);

	private readonly ConditionalWeakTable<object, Identity> _ids = new ConditionalWeakTable<object, Identity>();

	private long _nextId;
    internal bool TrackObjects;
    private readonly Dictionary<long, WeakReference<object>> _objects = new();
    internal object Resolve(long id) => _objects.TryGetValue(id, out var weak) && weak.TryGetTarget(out var obj)
        ? obj : throw new InvalidOperationException("Expired game object: " + id);

	private static readonly HashSet<string> ChoiceNodes = new HashSet<string>(StringComparer.Ordinal)
	{
		"NMapPoint", "NEventOptionButton", "NAncientDialogueLine", "NAncientDialogueHitbox", "NRestSiteButton", "NRewardButton", "NMerchantCard", "NMerchantRelic", "NMerchantPotion", "NMerchantCardRemoval",
		"NCardBundle", "NGridCardHolder", "NHandCardHolder", "NCardRewardSelectionScreen", "NChooseACardSelectionScreen", "NChooseABundleSelectionScreen", "NCardGridSelectionScreen", "NCardSelectScreen", "NDeckUpgradeSelectScreen", "NDeckTransformSelectScreen",
		"NDeckEnchantSelectScreen", "NPlayerHand", "NRewardsScreen", "NConfirmButton", "NProceedButton", "NChoiceSelectionSkipButton", "NCardRewardAlternativeButton", "NCrystalSphereScreen", "NTreasureRoomRelicHolder"
	};

	internal object? SelectionKey(SelectionOffer offer, PlayerChoiceResult result)
	{
		if (offer.Alternatives != null)
		{
			return new
			{
				command = "select_reward_option",
				args = new
				{
					index = result.AsIndexOrNull()
				}
			};
		}
		if (offer.Bundles != null)
		{
			return new
			{
				command = "select_bundle",
				args = new
				{
					index = result.AsIndexOrNull()
				}
			};
		}
		CardModel[] array = null;
		if (Read(result, "_indexes") is IEnumerable<int> source)
		{
			int[] source2 = source.ToArray();
			if (source2.Any((int i) => i < 0 || i >= offer.Cards.Length))
			{
				return null;
			}
			array = source2.Select((int i) => offer.Cards[i]).ToArray();
		}
		else
		{
			string[] array2 = new string[4] { "_canonicalCards", "_combatCards", "_deckCards", "_mutableCards" };
			foreach (string name in array2)
			{
				if (Read(result, name) is IEnumerable<CardModel> source3)
				{
					array = source3.ToArray();
					break;
				}
			}
		}
		if (array != null)
		{
			return new
			{
				command = "select_cards",
				args = new
				{
					card_instance_ids = array.Select(Id).ToArray()
				}
			};
		}
		return null;
	}

	internal object? ChoiceKey(object? source, string command, object[]? arguments = null)
	{
		if (command == "skip_treasure")
		{
			return Key("pick_relic", new
			{
				index = (int?)null
			});
		}
		if (command == "leave_shop" && source is MerchantRoom)
		{
			return Key("leave_shop", new { });
		}
		if (source is PlayCardAction { NetCombatCard: var netCombatCard } playCardAction)
		{
			CardModel cardModel = netCombatCard.ToCardModelOrNull();
			if (cardModel == null)
			{
				return null;
			}
			return Key("play_card", new
			{
				card_instance_id = Id(cardModel),
				target_instance_id = ((playCardAction.Target == null) ? ((long?)null) : new long?(Id(playCardAction.Target)))
			});
		}
		if (source is EndPlayerTurnAction)
		{
			return Key("end_turn", new { });
		}
		if (source is MoveToMapCoordAction obj && Read(obj, "_destination") is MapCoord mapCoord)
		{
			return Key("select_map_node", new { mapCoord.col, mapCoord.row });
		}
		if (source is CrystalSphereMinigame crystalSphereMinigame)
		{
			CrystalSphereCell crystalSphereCell = arguments?.OfType<CrystalSphereCell>().FirstOrDefault();
			if (crystalSphereCell != null)
			{
				return Key("crystal_sphere_cell", new
				{
					x = crystalSphereCell.X,
					y = crystalSphereCell.Y,
					tool = crystalSphereMinigame.CrystalSphereTool.ToString()
				});
			}
		}
		UsePotionAction use = source as UsePotionAction;
		if (use != null)
		{
			int num = checked((int)use.PotionIndex);
			PotionModel potionAtSlotIndex = use.Player.GetPotionAtSlotIndex(num);
			if (potionAtSlotIndex == null)
			{
				return null;
			}
			Creature creature = null;
			if (CombatManager.Instance.IsInProgress)
			{
				if (!use.TargetId.HasValue && potionAtSlotIndex.TargetType.IsSingleTarget())
				{
					creature = use.Player.Creature;
				}
				else if (use.TargetId.HasValue)
				{
					CombatState combatState = CombatManager.Instance.DebugOnlyGetState();
					creature = combatState?.PlayerCreatures.Concat(combatState.Enemies).SingleOrDefault((Creature c) => c.CombatId == use.TargetId);
					if (creature == null)
					{
						return null;
					}
				}
			}
			else
			{
				object obj2 = Read(use, "TargetPlayerId");
				if (obj2 is ulong)
				{
					ulong targetPlayerId = (ulong)obj2;
					creature = RunManager.Instance.DebugOnlyGetState()?.Players.SingleOrDefault((Player p) => p.NetId == targetPlayerId)?.Creature;
					if (creature == null)
					{
						return null;
					}
				}
				else if (potionAtSlotIndex.TargetType != TargetType.TargetedNoCreature)
				{
					creature = use.Player.Creature;
				}
			}
			return Key("use_potion", new
			{
				potion_instance_id = Id(potionAtSlotIndex),
				slot = num,
				target_instance_id = ((creature == null) ? ((long?)null) : new long?(Id(creature)))
			});
		}
		if (source is DiscardPotionGameAction obj3)
		{
			int num2 = Convert.ToInt32(Read(obj3, "_potionSlotIndex"));
			PotionModel potionModel = RunManager.Instance.DebugOnlyGetState()?.Players.Single().GetPotionAtSlotIndex(num2);
			if (potionModel != null)
			{
				return Key("discard_potion", new
				{
					potion_instance_id = Id(potionModel),
					slot = num2
				});
			}
			return null;
		}
		if (source is PickRelicAction obj4)
		{
			return Key("pick_relic", new
			{
				index = (int?)Read(obj4, "_relicIndex")
			});
		}
		if (source is EventOption obj5)
		{
			return Key("choose_event_option", new
			{
				option_instance_id = Id(obj5)
			});
		}
		if (command == "choose_rest_option" && arguments != null && arguments.Length > 0)
		{
			return Key(command, new
			{
				index = Convert.ToInt32(arguments[0])
			});
		}
		if (source is MerchantEntry obj6)
		{
			return Key("purchase", new
			{
				entry_instance_id = Id(obj6)
			});
		}
		if (command == "take_reward")
		{
			Reward reward = arguments?.OfType<Reward>().FirstOrDefault();
			if (reward != null)
			{
				return Key(command, new
				{
					reward_instance_id = Id(reward)
				});
			}
		}
		if (command == "skip_rewards" && Read(arguments?.FirstOrDefault(), "set") is RewardsSet rewardsSet)
		{
			return Key(command, new
			{
				reward_set_id = rewardsSet.Id
			});
		}
		return null;
		static object Key(string name, object args)
		{
			return new
			{
				command = name,
				args = args
			};
		}
	}

	internal static object MatchLegal(JsonElement state, JsonElement key)
	{
		if (key.ValueKind != JsonValueKind.Object)
		{
			return new
			{
				status = "unmapped",
				candidate_index = (int?)null
			};
		}
		if (!state.TryGetProperty("legal", out var value) || !value.TryGetProperty("status", out var value2) || value2.GetString() != "complete")
		{
			return new
			{
				status = "unavailable",
				candidate_index = (int?)null
			};
		}
		if (key.GetProperty("command").GetString() == "select_cards" && value.TryGetProperty("selection", out var value3) && value3.ValueKind == JsonValueKind.Object)
		{
			long[] array = (from v in key.GetProperty("args").GetProperty("card_instance_ids").EnumerateArray()
				select v.GetInt64()).ToArray();
			JsonElement[] candidates = value.GetProperty("actions").EnumerateArray().ToArray();
			int[] array2 = array.Select((long id) => Array.FindIndex(candidates, (JsonElement c) => c.GetProperty("command").GetString() == "select_card" && c.GetProperty("args").GetProperty("card_instance_id").GetInt64() == id)).ToArray();
			return new
			{
				status = ((((array.Length == 0 && value3.GetProperty("cancelable").GetBoolean()) || (array.Length >= value3.GetProperty("min").GetInt32() && array.Length <= value3.GetProperty("max").GetInt32())) && array.Distinct().Count() == array.Length && array2.All((int i) => i >= 0)) ? "matched" : "invalid_selection"),
				candidate_index = (int?)null,
				candidate_indices = array2
			};
		}
		string target = Signature(key);
		(JsonElement, int)[] array3 = (from p in value.GetProperty("actions").EnumerateArray().Select((JsonElement candidate, int index) => (candidate: candidate, index: index))
			where Signature(p.candidate) == target
			select p).ToArray();
		return new
		{
			status = ((array3.Length == 1) ? "matched" : ((array3.Length == 0) ? "not_in_candidates" : "ambiguous")),
			candidate_index = ((array3.Length == 1) ? new int?(array3[0].Item2) : ((int?)null))
		};
		static string Signature(JsonElement item)
		{
			return item.GetProperty("command").GetString() + ":" + string.Join("|", from p in item.GetProperty("args").EnumerateObject()
				orderby p.Name
				select p.Name + "=" + p.Value.GetRawText());
		}
	}

	internal object LegalActions(RunState run, bool ended, string? decision, object? source = null, object[]? arguments = null)
	{
		List<object> actions = new List<object>();
		string status = "unavailable";
		string scope = "unknown";
		string reason = "decision_boundary_not_identified";
		object context = null;
		object selection = null;
		if (ended || run.IsGameOver)
		{
			status = "complete";
			scope = "terminal";
			reason = null;
			return Result();
		}
		try
		{
			CombatManager instance = CombatManager.Instance;
			CombatState combatState = instance?.DebugOnlyGetState();
			Player player = run.Players.Single();
			PlayerCombatState playerCombatState = player.PlayerCombatState;
			bool flag = instance != null && instance.IsInProgress && combatState != null && playerCombatState != null;
			bool flag2;
			switch (decision)
			{
			case "PlayCardAction":
			case "UsePotionAction":
			case "EndPlayerTurnAction":
			case "DiscardPotionGameAction":
				flag2 = true;
				break;
			default:
				flag2 = false;
				break;
			}
			bool flag3 = flag2;
			flag2 = !flag;
			if (flag2)
			{
				bool flag4 = ((decision == "UsePotionAction" || decision == "DiscardPotionGameAction") ? true : false);
				flag2 = flag4;
			}
			if (flag2)
			{
				IOverlayScreen overlayScreen = NOverlayStack.Instance?.Peek();
				NMapScreen instance2 = NMapScreen.Instance;
				if (overlayScreen != null && overlayScreen.GetType().Name != "NRewardsScreen")
				{
					reason = "potion_overlay_not_adapted:" + overlayScreen.GetType().Name;
					return Result();
				}
				if (instance2 != null && instance2.IsOpen && instance2.IsTravelEnabled)
				{
					decision = "MoveToMapCoordAction";
				}
				else
				{
					RewardsSet rewardsSet = ActiveRewardsSet(run, null);
					if (rewardsSet != null && rewardsSet.Rewards.Any((Reward r) => !r.SuccessfullySelected))
					{
						decision = "take_reward";
						arguments = new object[1] { rewardsSet.Rewards.First((Reward r) => !r.SuccessfullySelected) };
					}
					else
					{
						if (overlayScreen != null)
						{
							reason = "potion_reward_surface_without_active_set";
							return Result();
						}
						if (run.CurrentRoom is MerchantRoom merchantRoom)
						{
							decision = "leave_shop";
							source = merchantRoom;
						}
						else if (run.CurrentRoom?.RoomType.ToString() == "RestSite")
						{
							decision = "choose_rest_option";
						}
						else
						{
							EventModel eventModel = CurrentEvent(run);
							if (eventModel != null && !eventModel.IsFinished)
							{
								decision = "choose_event_option";
							}
							else
							{
								if (!(run.CurrentRoom?.RoomType.ToString() == "Treasure"))
								{
									reason = "potion_strategic_surface_not_identified";
									return Result();
								}
								decision = "PickRelicAction";
							}
						}
					}
				}
			}
			if (decision == "card_selection" && source is SelectionOffer selectionOffer)
			{
				scope = ((selectionOffer.Bundles == null) ? "card_select" : "bundle_select");
				if (selectionOffer.Alternatives != null)
				{
					scope = "card_reward";
					context = new
					{
						cards = selectionOffer.Cards.Select(Card).ToArray(),
						alternatives = selectionOffer.Alternatives.Select((CardRewardAlternative a) => Value(a)).ToArray()
					};
					for (int num = 0; num < selectionOffer.Cards.Length + selectionOffer.Alternatives.Length; num++)
					{
						Add("select_reward_option", new
						{
							index = (int?)num
						});
					}
					if (selectionOffer.Cancelable)
					{
						Add("select_reward_option", new
						{
							index = (int?)null
						});
					}
				}
				else if (selectionOffer.Bundles != null)
				{
					context = selectionOffer.Bundles.Select((CardModel[] source3, int index) => new
					{
						index = index,
						cards = source3.Select(Card).ToArray()
					}).ToArray();
					for (int num2 = 0; num2 < selectionOffer.Bundles.Length; num2++)
					{
						Add("select_bundle", new
						{
							index = (int?)num2
						});
					}
				}
				else
				{
					if (selectionOffer.Cards.Distinct(ReferenceEqualityComparer.Instance).Count() != selectionOffer.Cards.Length)
					{
						reason = "duplicate_card_references_in_offer";
						return Result();
					}
					context = selectionOffer.Cards.Select(Card).ToArray();
					selection = new
					{
						min = selectionOffer.Min,
						max = selectionOffer.Max,
						cancelable = selectionOffer.Cancelable,
						ordered = true
					};
					CardModel[] cards = selectionOffer.Cards;
					foreach (CardModel obj in cards)
					{
						Add("select_card", new
						{
							card_instance_id = Id(obj)
						});
					}
				}
				status = "complete";
				reason = null;
				return Result();
			}
			if (flag && flag3)
			{
				scope = "combat_play";
				if (playerCombatState.Phase != PlayerTurnPhase.Play)
				{
					reason = "not_player_play_phase";
					return Result();
				}
				foreach (CardModel card in playerCombatState.Hand.Cards)
				{
					if (!card.CanPlay(out UnplayableReason _, out AbstractModel _))
					{
						continue;
					}
					if (card.IsValidTarget(null))
					{
						Add("play_card", new
						{
							card_instance_id = Id(card),
							target_instance_id = (long?)null
						});
					}
					foreach (Creature item2 in combatState.PlayerCreatures.Concat(combatState.Enemies))
					{
						if (card.IsValidTarget(item2))
						{
							Add("play_card", new
							{
								card_instance_id = Id(card),
								target_instance_id = (long?)Id(item2)
							});
						}
					}
				}
				Add("end_turn", new { });
			}
			else
			{
				if (!flag && decision == "choose_event_option")
				{
					EventModel eventModel2 = CurrentEvent(run);
					if (eventModel2 != null)
					{
						scope = "event_choice";
						context = new
						{
							event_id = eventModel2.Id.Entry,
							options = eventModel2.CurrentOptions.Select(Option).ToArray()
						};
						foreach (EventOption currentOption in eventModel2.CurrentOptions)
						{
							if (IsFinalEventOption(currentOption))
							{
								Add("choose_event_option", new
								{
									option_instance_id = Id(currentOption)
								});
							}
						}
						goto IL_0e12;
					}
				}
				if (!flag && decision == "choose_rest_option")
				{
					scope = "rest_site";
					IReadOnlyList<RestSiteOption> localOptions = RunManager.Instance.RestSiteSynchronizer.GetLocalOptions();
					if (localOptions == null || localOptions.Count == 0)
					{
						return Result();
					}
					context = localOptions.Select((RestSiteOption option, int index) => new
					{
						index = index,
						instance_id = Id(option),
						option_id = option.OptionId,
						enabled = option.IsEnabled,
						title = option.Title?.ToString(),
						description = option.Description?.ToString()
					}).ToArray();
					for (int num4 = 0; num4 < localOptions.Count; num4++)
					{
						if (localOptions[num4].IsEnabled)
						{
							Add("choose_rest_option", new
							{
								index = num4
							});
						}
					}
				}
				else if (!flag && decision == "MoveToMapCoordAction")
				{
					scope = "map_select";
					if (run.Map == null)
					{
						reason = "map_not_ready";
						return Result();
					}
					IEnumerable<MapPoint> source2;
					if (run.VisitedMapCoords.Count == 0)
					{
						source2 = new[] { run.Map.StartingMapPoint };
					}
					else
					{
						IReadOnlyList<MapCoord> visitedMapCoords = run.VisitedMapCoords;
						MapCoord mapCoord = visitedMapCoords[visitedMapCoords.Count - 1];
						if (mapCoord == run.Map.BossMapPoint.coord && Read(run.Map, "SecondBossMapPoint") is MapPoint item)
						{
							source2 = new[] { item };
						}
						else if (mapCoord.row == run.Map.GetRowCount() - 1)
						{
							source2 = new[] { run.Map.BossMapPoint };
						}
						else
						{
							MapPoint point = run.Map.GetPoint(mapCoord);
							if (point == null)
							{
								reason = "last_visited_map_point_missing";
								return Result();
							}
							source2 = MapTravel.GetTravelablePointsFrom(run, point);
						}
					}
					MapPoint[] array = source2.Distinct().ToArray();
					context = array.Select((MapPoint p) => new
					{
						col = p.coord.col,
						row = p.coord.row,
						type = p.PointType.ToString()
					}).ToArray();
					MapPoint[] array2 = array;
					foreach (MapPoint mapPoint in array2)
					{
						Add("select_map_node", new
						{
							mapPoint.coord.col,
							mapPoint.coord.row
						});
					}
				}
				else if (!flag && decision == "crystal_sphere_cell" && source is CrystalSphereMinigame crystalSphereMinigame)
				{
					scope = "crystal_sphere";
					context = new
					{
						remaining = crystalSphereMinigame.DivinationCount,
						width = crystalSphereMinigame.GridSize.X,
						height = crystalSphereMinigame.GridSize.Y,
						cells = (from CrystalSphereCell c in crystalSphereMinigame.cells
							select Value(c)).ToArray()
					};
					if (!crystalSphereMinigame.IsFinished && crystalSphereMinigame.DivinationCount > 0)
					{
						CrystalSphereCell[,] cells = crystalSphereMinigame.cells;
						foreach (CrystalSphereCell crystalSphereCell in cells)
						{
							if (crystalSphereCell.IsHidden)
							{
								CrystalSphereMinigame.CrystalSphereToolType[] array3 = new CrystalSphereMinigame.CrystalSphereToolType[2]
								{
									CrystalSphereMinigame.CrystalSphereToolType.Small,
									CrystalSphereMinigame.CrystalSphereToolType.Big
								};
								for (int num7 = 0; num7 < array3.Length; num7++)
								{
									CrystalSphereMinigame.CrystalSphereToolType crystalSphereToolType = array3[num7];
									Add("crystal_sphere_cell", new
									{
										x = crystalSphereCell.X,
										y = crystalSphereCell.Y,
										tool = crystalSphereToolType.ToString()
									});
								}
							}
						}
					}
				}
				else
				{
					flag2 = !flag;
					if (flag2)
					{
						bool flag4 = ((decision == "purchase" || decision == "leave_shop") ? true : false);
						flag2 = flag4;
					}
					if (flag2)
					{
						MerchantInventory merchantInventory = arguments?.OfType<MerchantInventory>().FirstOrDefault() ?? (source as MerchantRoom)?.GetLocalInventory();
						if (merchantInventory != null)
						{
							scope = "shop";
							MerchantEntry[] array4 = merchantInventory.AllEntries.ToArray();
							context = array4.Select((MerchantEntry entry) => new
							{
								instance_id = Id(entry),
								value = Value(entry)
							}).ToArray();
							MerchantEntry[] array5 = array4;
							foreach (MerchantEntry merchantEntry in array5)
							{
								if (merchantEntry.IsStocked && merchantEntry.EnoughGold
                                    && (merchantEntry is not MerchantPotionEntry { Model: { } potion } || (player.HasOpenPotionSlots
                                        && Hook.ShouldProcurePotion(run, player.Creature.CombatState, potion, player)))
                                    && (merchantEntry is not MerchantCardRemovalEntry || player.Deck.Cards.Any(c => c.IsRemovable)))
								{
									Add("purchase", new
									{
										entry_instance_id = Id(merchantEntry)
									});
								}
							}
							Add("leave_shop", new { });
							goto IL_0e12;
						}
					}
					flag2 = !flag;
					if (flag2)
					{
						bool flag4 = ((decision == "PickRelicAction" || decision == "skip_treasure") ? true : false);
						flag2 = flag4;
					}
					if (flag2)
					{
						scope = "treasure";
						IReadOnlyList<RelicModel> currentRelics = RunManager.Instance.TreasureRoomRelicSynchronizer.CurrentRelics;
						if (currentRelics == null)
						{
							reason = "treasure_not_ready";
							return Result();
						}
						context = currentRelics.Select((RelicModel relic, int index) => new
						{
							index = index,
							relic = Value(relic)
						}).ToArray();
						for (int num8 = 0; num8 < currentRelics.Count; num8++)
						{
							Add("pick_relic", new
							{
								index = (int?)num8
							});
						}
						Add("pick_relic", new
						{
							index = (int?)null
						});
					}
					else
					{
						if ((!(decision == "take_reward") && !(decision == "skip_rewards")) || 1 == 0)
						{
							return Result();
						}
						scope = "reward_choice";
						RewardsSet rewardsSet2 = ((decision == "skip_rewards") ? (Read(arguments?.FirstOrDefault(), "set") as RewardsSet) : ActiveRewardsSet(run, arguments?.OfType<Reward>().FirstOrDefault()));
						if (rewardsSet2 == null)
						{
							reason = "active_reward_set_unavailable";
							return Result();
						}
						context = new
						{
							reward_set_id = rewardsSet2.Id,
							disallow_skipping = rewardsSet2.DisallowSkipping,
							rewards = rewardsSet2.Rewards.Select(Reward).ToArray()
						};
						foreach (Reward reward in rewardsSet2.Rewards)
						{
							if (!reward.SuccessfullySelected
                                && (reward is not PotionReward { Potion: { } potion } || (player.HasOpenPotionSlots
                                    && Hook.ShouldProcurePotion(run, player.Creature.CombatState, potion, player))))
							{
								Add("take_reward", new
								{
									reward_instance_id = Id(reward)
								});
							}
						}
						if (!rewardsSet2.DisallowSkipping)
						{
							Add("skip_rewards", new
							{
								reward_set_id = rewardsSet2.Id
							});
						}
					}
				}
			}
			goto IL_0e12;
			IL_0e12:
			if (player.CanUseOrRemovePotions)
			{
				for (int num9 = 0; num9 < player.MaxPotionCount; num9++)
				{
					PotionModel potionAtSlotIndex = player.GetPotionAtSlotIndex(num9);
					if (potionAtSlotIndex == null)
					{
						continue;
					}
					Add("discard_potion", new
					{
						potion_instance_id = Id(potionAtSlotIndex),
						slot = num9
					});
					if (potionAtSlotIndex.IsQueued || !potionAtSlotIndex.PassesCustomUsabilityCheck || (!(potionAtSlotIndex.Usage.ToString() == "AnyTime") && !(potionAtSlotIndex.Usage.ToString() == "CombatOnly" && flag)))
					{
						continue;
					}
					List<Creature> list = new List<Creature> { null, player.Creature };
					if (flag)
					{
						list.AddRange(combatState.PlayerCreatures.Concat(combatState.Enemies));
					}
					foreach (Creature item3 in list.Distinct())
					{
						if (potionAtSlotIndex.IsValidTarget(item3))
						{
							Add("use_potion", new
							{
								potion_instance_id = Id(potionAtSlotIndex),
								slot = num9,
								target_instance_id = ((item3 == null) ? ((long?)null) : new long?(Id(item3)))
							});
						}
					}
				}
			}
			status = "complete";
			reason = null;
		}
		catch (Exception ex)
		{
			actions.Clear();
			status = "error";
			reason = ex.ToString();
		}
		return Result();
		void Add(string command, object args)
		{
			actions.Add(new { command, args });
		}
		object Result()
		{
			return new
			{
				schema = 1,
				status = status,
				scope = scope,
				reason = reason,
				context = context,
				selection = selection,
				actions = actions.ToArray()
			};
		}
	}

	internal static RewardsSet? ActiveRewardsSet(RunState run, Reward? reward)
	{
		if (run.Players.Count != 1)
		{
			return null;
		}
		if (!(Read(RunManager.Instance.RewardsSetSynchronizer, "_rewardStates") is IEnumerable enumerable))
		{
			return null;
		}
		foreach (object item in enumerable)
		{
			if (!(Read(item, "rewardsStack") is IEnumerable source))
			{
				continue;
			}
			foreach (object item2 in source.Cast<object>().Reverse())
			{
				if (Read(item2, "set") is RewardsSet rewardsSet && rewardsSet.Player == run.Players[0] && (reward == null || rewardsSet.Rewards.Contains(reward)))
				{
					return rewardsSet;
				}
			}
		}
		return null;
	}

	public long Id(object obj)
	{
		var id = _ids.GetValue(obj, (object _) => new Identity(++_nextId)).Id;
        if (TrackObjects) _objects[id] = new WeakReference<object>(obj);
        return id;
	}

	internal static object? Read(object? obj, string name)
	{
		if (obj == null)
		{
			return null;
		}
		Type type = obj.GetType();
		while (type != null)
		{
			FieldInfo field = type.GetField(name, BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
			if (field != null)
			{
				return field.GetValue(obj);
			}
			PropertyInfo property = type.GetProperty(name, BindingFlags.DeclaredOnly | BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
			if ((object)property != null && property.GetIndexParameters().Length == 0)
			{
				return property.GetValue(obj);
			}
			type = type.BaseType;
		}
		return null;
	}

	public object Capture(RunState run, bool ended = false, bool? victory = null, string? decision = null, object? source = null, object[]? arguments = null)
	{
		CombatState combatState = CombatManager.Instance?.DebugOnlyGetState();
		bool flag = CombatManager.Instance?.IsInProgress ?? false;
		EventModel eventModel = CurrentEvent(run);
		return new
		{
			run = new
			{
				seed = run.Rng.StringSeed,
				ascension = run.AscensionLevel,
				act = run.CurrentActIndex + 1,
				floor = run.ActFloor,
				total_floor = run.TotalFloor,
				location = Value(run.CurrentMapCoord),
				room = run.CurrentRoom?.RoomType.ToString(),
				room_id = Value(Read(run.CurrentRoom, "ModelId") ?? Read(Read(run.CurrentRoom, "Model"), "Id") ?? Read(Read(run.CurrentRoom, "Encounter"), "Id")),
				game_over = (ended || run.IsGameOver),
				engine_game_over = run.IsGameOver,
				ended = ended,
				victory = victory,
				abandoned = RunManager.Instance.IsAbandoned,
				win_time = RunManager.Instance.WinTime,
				boss = run.Acts.ElementAtOrDefault(run.CurrentActIndex)?.BossEncounter?.Id.Entry,
				visited = Value(run.VisitedMapCoords)
			},
			domain = (flag ? "combat" : "strategic"),
			legal = LegalActions(run, ended, decision, source, arguments),
			current_event = ((eventModel == null) ? null : Event(eventModel)),
			event_state_ready = (eventModel != null),
			players = run.Players.Select(PlayerState).ToArray(),
			combat = ((combatState == null) ? null : new
			{
				instance_id = Id(combatState),
				in_progress = flag,
				round = combatState.RoundNumber,
				actions_disabled = CombatManager.Instance?.PlayerActionsDisabled,
				enemies = combatState.Enemies.Select(CreatureState).ToArray()
			}),
			map = (from p in (run.Map?.GetAllMapPoints() ?? Array.Empty<MapPoint>())
                .Concat(run.Map == null ? Array.Empty<MapPoint>() : new[] { run.Map.StartingMapPoint, run.Map.BossMapPoint })
                .Concat(run.Map?.SecondBossMapPoint is { } second ? new[] { second } : Array.Empty<MapPoint>()).Distinct()
				select new
				{
					col = p.coord.col,
					row = p.coord.row,
					type = p.PointType.ToString(),
					children = p.Children.Select((MapPoint c) => new
					{
						c.coord.col,
						c.coord.row
					}).ToArray()
				}).ToArray(),
			ui = VisibleChoices(),
			executor = new
			{
				running = RunManager.Instance.ActionExecutor?.IsRunning,
				current_action_id = RunManager.Instance.ActionExecutor?.CurrentlyRunningAction?.Id
			},
			visibility = new
			{
				draw_pile_order = "engine_internal",
				rng_state_included = false
			}
		};
	}

	internal static EventModel? CurrentEvent(RunState run)
	{
		if (run.CurrentRoom?.RoomType.ToString() != "Event")
		{
			return null;
		}
		IReadOnlyList<EventModel> readOnlyList = RunManager.Instance.EventSynchronizer?.Events;
		if (readOnlyList == null || readOnlyList.Count <= 0)
		{
			return null;
		}
		return readOnlyList[0];
	}

	public object Event(EventModel model)
	{
		return new
		{
			instance_id = Id(model),
			id = model.Id.Entry,
			type = model.GetType().FullName,
			description = model.Description?.ToString(),
			finished = model.IsFinished,
			vars = Variables(model),
			options = model.CurrentOptions.Select(Option).ToArray(),
			chosen_card_type = ((!(model is TinkerTime)) ? null : Read(model, "ChosenCardType")?.ToString())
		};
	}

	public object Option(EventOption option)
	{
		return new
		{
			instance_id = Id(option),
			text_key = option.TextKey,
			title = option.Title?.ToString(),
			description = option.Description?.ToString(),
			locked = option.IsLocked,
			proceed = option.IsProceed,
			was_chosen = option.WasChosen,
			disable_on_chosen = Read(option, "DisableOnChosen"),
			rider_effect = Rider(option.TextKey),
			relic = Value(option.Relic)
		};
	}

	internal static bool IsFinalEventOption(EventOption option)
	{
		if (!option.IsLocked && !option.IsProceed && Read(option, "OnChosen") is Delegate)
		{
			object obj = Read(option, "DisableOnChosen");
			if (obj is bool)
			{
				if ((bool)obj)
				{
					return !option.WasChosen;
				}
				return true;
			}
		}
		return false;
	}

	private static string? Rider(string key)
	{
		if (!key.StartsWith("TINKER_TIME.pages.CHOOSE_RIDER.options.", StringComparison.Ordinal) || !Enum.TryParse<TinkerTime.RiderEffect>(key.Split('.').Last(), ignoreCase: true, out var result))
		{
			return null;
		}
		return result.ToString();
	}

	private object PlayerState(Player player)
	{
		PlayerCombatState playerCombatState = player.PlayerCombatState;
		return new
		{
			character = player.Character.Id.Entry,
			creature = CreatureState(player.Creature),
			gold = player.Gold,
			deck = player.Deck.Cards.Select(Card).ToArray(),
			relics = player.Relics.Select(Model).ToArray(),
			potions = (from slot in Enumerable.Range(0, player.MaxPotionCount)
				select new
				{
					slot = slot,
					potion = Value(player.GetPotionAtSlotIndex(slot))
				}).ToArray(),
			max_potions = player.MaxPotionCount,
			combat = ((playerCombatState == null) ? null : new
			{
				phase = playerCombatState.Phase.ToString(),
				energy = playerCombatState.Energy,
				max_energy = playerCombatState.MaxEnergy,
				stars = playerCombatState.Stars,
				hand = playerCombatState.Hand.Cards.Select((CardModel c) => Card(c, inHand: true)).ToArray(),
				draw = playerCombatState.DrawPile.Cards.Select(Card).ToArray(),
				discard = playerCombatState.DiscardPile.Cards.Select(Card).ToArray(),
				exhaust = playerCombatState.ExhaustPile.Cards.Select(Card).ToArray(),
				play = Value(Read(playerCombatState, "PlayPile")),
				orbs = Value(Read(playerCombatState.OrbQueue, "Orbs")),
				orb_slots = Read(playerCombatState.OrbQueue, "Capacity")
			}),
			osty = ((player.Osty == null) ? null : CreatureState(player.Osty))
		};
	}

	public object Card(CardModel card)
	{
		return Card(card, inHand: false);
	}

	private object Card(CardModel card, bool inHand)
	{
		return new
		{
			instance_id = Id(card),
			id = card.Id.Entry,
			type = card.GetType().FullName,
			upgraded = card.IsUpgraded,
			upgrade_level = card.CurrentUpgradeLevel,
			cost = Read(card.EnergyCost, "Canonical"),
			current_cost = card.EnergyCost?.GetResolved(),
			costs_x = card.EnergyCost?.CostsX,
			target_type = card.TargetType.ToString(),
			card_type = card.Type.ToString(),
			rarity = card.Rarity.ToString(),
			keywords = Value(Read(card, "Keywords")),
			vars = Variables(card),
			enchantment = Value(Read(card, "Enchantment")),
			affliction = Value(Read(card, "Affliction")),
			star_cost = (card.IsCanonical ? card.CanonicalStarCost : card.GetStarCostWithModifiers()),
			stars_x = card.HasStarCostX,
			retain = (!card.IsCanonical && card.ShouldRetainThisTurn),
			exhaust_on_next_play = card.ExhaustOnNextPlay,
			saved_properties = Saved(card),
			mad_science = ((card is MadScience { TinkerTimeType: var tinkerTimeType } madScience) ? new
			{
				card_type = tinkerTimeType.ToString(),
				rider_effect = madScience.TinkerTimeRider.ToString(),
				rider_effect_id = (int)madScience.TinkerTimeRider
			} : null),
			playability = (inHand ? Playability(card) : null)
		};
	}

	private object Playability(CardModel card)
	{
		CombatManager instance = CombatManager.Instance;
		if (instance == null || !instance.IsInProgress || card.Owner?.PlayerCombatState == null)
		{
			return new
			{
				evaluated = false,
				reason = "not_in_combat"
			};
		}
		UnplayableReason reason;
		AbstractModel preventer;
		bool flag = card.CanPlay(out reason, out preventer);
		var array = (from c in instance.DebugOnlyGetState().PlayerCreatures.Concat(instance.DebugOnlyGetState().Enemies).Where(card.IsValidTarget)
			select new
			{
				instance_id = Id(c),
				combat_id = Read(c, "CombatId")
			}).ToArray();
		return new
		{
			evaluated = true,
			can_play = (flag && (card.IsValidTarget(null) || array.Length != 0)),
			card_can_play = flag,
			unplayable_reason = reason.ToString(),
			prevented_by = preventer?.Id.Entry,
			valid_targets = array,
			accepts_null_target = card.IsValidTarget(null),
			player_phase = card.Owner.PlayerCombatState.Phase.ToString(),
			actions_disabled = instance.PlayerActionsDisabled
		};
	}

	private static object? Saved(AbstractModel model)
	{
		if (!model.IsCanonical)
		{
			return Journal.Freeze(SavedProperties.From(model));
		}
		return null;
	}

	private object CreatureState(Creature c)
	{
		return new
		{
			instance_id = Id(c),
			combat_id = Read(c, "CombatId"),
			id = (c.Monster?.Id.Entry ?? c.ModelId.Entry),
			hp = c.CurrentHp,
			max_hp = c.MaxHp,
			block = c.Block,
			alive = c.IsAlive,
			powers = c.Powers.Select(Model).ToArray(),
			move = Read(c.Monster?.NextMove, "Id"),
			intents = (c.Monster?.NextMove?.Intents ?? Array.Empty<AbstractIntent>()).Select((AbstractIntent i) => new
			{
				type = i.IntentType.ToString(),
				repeats = Read(i, "Repeats"),
				damage = IntentDamage(i, c)
			}).ToArray()
		};
	}

	private static object? IntentDamage(object intent, Creature creature)
	{
		if (!(intent is AttackIntent attackIntent))
		{
			return null;
		}
		List<Creature> list = CombatManager.Instance?.DebugOnlyGetState()?.PlayerCreatures?.ToList();
		if (list != null)
		{
			return attackIntent.GetSingleDamage(list, creature);
		}
		return null;
	}

	private object Model(AbstractModel model)
	{
		return new
		{
			instance_id = Id(model),
			id = model.Id.Entry,
			type = model.GetType().FullName,
			title = Read(model, "Title")?.ToString(),
			description = Read(model, "Description")?.ToString(),
			amount = Read(model, "Amount"),
			counter = Read(model, "DisplayAmount"),
			status = Read(model, "Status")?.ToString(),
			vars = Variables(model),
			saved_properties = Saved(model),
			passive_value = ((model is OrbModel orbModel) ? new decimal?(orbModel.PassiveVal) : ((decimal?)null)),
			evoke_value = ((model is OrbModel orbModel2) ? new decimal?(orbModel2.EvokeVal) : ((decimal?)null))
		};
	}

	private static object Variables(object model)
	{
		Dictionary<string, object> dictionary = new Dictionary<string, object>();
		if (!(Read(model, "DynamicVars") is IEnumerable enumerable))
		{
			return dictionary;
		}
		foreach (object item in enumerable)
		{
			object obj = Read(item, "Value") ?? item;
			string text = Read(item, "Key")?.ToString() ?? Read(obj, "Name")?.ToString();
			if (text != null)
			{
				dictionary[text] = new
				{
					base_value = Read(obj, "BaseValue"),
					value = Read(obj, "IntValue")
				};
			}
		}
		return dictionary;
	}

	public object? Value(object? value, int depth = 0)
	{
		if (value == null || value is string || value is bool || value is decimal || value.GetType().IsPrimitive)
		{
			return value;
		}
		if (value is Enum)
		{
			return value.ToString();
		}
		if (value is CardModel card)
		{
			return Card(card);
		}
		if (value is EventOption option)
		{
			return Option(option);
		}
		if (value is EventModel model)
		{
			return Event(model);
		}
		if (value is Reward reward)
		{
			return Reward(reward);
		}
		if (value is Player player)
		{
			return new
			{
				character = player.Character.Id.Entry
			};
		}
		if (value is Creature c)
		{
			return CreatureState(c);
		}
		if (value is ModelId modelId)
		{
			return modelId.Entry;
		}
		if (value is AbstractModel model2)
		{
			// Orb state is live and instance-specific. In particular DarkOrb.EvokeVal
			// is accumulated damage, not its canonical value or passive gain.
			// Do not depend on localized descriptions or save-property serialization.
			if (model2 is OrbModel orb)
				return new { instance_id = Id(orb), id = orb.Id.Entry,
					type = orb.GetType().FullName, passive_value = orb.PassiveVal,
					evoke_value = orb.EvokeVal };
			return Model(model2);
		}
		if (value is CardPile cardPile)
		{
			return cardPile.Cards.Select(Card).ToArray();
		}
		if (value is CrystalSphereCell { X: var x, Y: var y } crystalSphereCell)
		{
			return new
			{
				x = x,
				y = y,
				hidden = crystalSphereCell.IsHidden,
				item = (crystalSphereCell.IsHidden ? null : Value(crystalSphereCell.Item, depth + 1))
			};
		}
		if (depth >= 6)
		{
			return new
			{
				type = value.GetType().FullName,
				truncated = true
			};
		}
		if (value is IEnumerable enumerable)
		{
			List<object> list = new List<object>();
			foreach (object item in enumerable)
			{
				if (list.Count == 1024)
				{
					list.Add(new
					{
						truncated = true
					});
					break;
				}
				list.Add(Value(item, depth + 1));
			}
			return list;
		}
		Dictionary<string, object> dictionary = new Dictionary<string, object> { ["type"] = value.GetType().FullName };
		string[] array;
		if (value is GodotObject godotObject)
		{
			dictionary["node_id"] = Id(godotObject);
			dictionary["parent_type"] = (godotObject as Node)?.GetParent()?.GetType().Name;
			array = new string[8] { "CardModel", "Option", "Reward", "Entry", "Bundle", "Point", "_line", "_entity" };
			foreach (string text in array)
			{
				object obj = Read(value, text);
				if (obj != null)
				{
					dictionary[text] = Value(obj, depth + 1);
				}
			}
			return dictionary;
		}
		FieldInfo[] fields = value.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public);
		foreach (FieldInfo fieldInfo in fields)
		{
			dictionary[fieldInfo.Name] = Value(fieldInfo.GetValue(value), depth + 1);
		}
		array = new string[33]
		{
			"Id", "Index", "Card", "Model", "CreationResult", "Cost", "IsStocked", "OptionId", "TextKey", "Title",
			"Description", "IsLocked", "IsEnabled", "Potion", "MinSelect", "MaxSelect", "Cancelable", "Destination", "Coord", "coord",
			"PointType", "PassiveVal", "EvokeVal", "Type", "SelectedIndex", "Text", "CombatCardIndex", "DeckCardIndex", "CrystalSphereTool", "DivinationCount",
			"GridSize", "IsFinished", "IsOnSale"
		};
		foreach (string text2 in array)
		{
			object obj2 = Read(value, text2);
			if (obj2 != null)
			{
				string? fullName = obj2.GetType().FullName;
				dictionary[text2] = ((fullName != null && fullName.Contains("LocString")) ? obj2.ToString() : Value(obj2, depth + 1));
			}
		}
		return dictionary;
	}

	public object Reward(Reward reward)
	{
		return new
		{
			instance_id = Id(reward),
			type = reward.GetType().FullName,
			reward_type = Read(reward, "RewardType")?.ToString(),
			selected = reward.SuccessfullySelected,
			description = reward.Description?.ToString(),
			cards = ((reward is CardReward { IsPopulated: not false } cardReward) ? cardReward.Cards.Select(Card).ToArray() : null),
			value = Value(Read(reward, "Amount") ?? Read(reward, "Gold") ?? Read(reward, "Relic") ?? Read(reward, "Potion"))
		};
	}

	public object Action(GameAction action)
	{
		CardModel cardModel = ((action is PlayCardAction { NetCombatCard: var netCombatCard }) ? netCombatCard.ToCardModelOrNull() : (Read(action, "_card") as CardModel));
		IReadOnlyList<CardModel> readOnlyList = (RunManager.Instance.DebugOnlyGetState()?.Players.FirstOrDefault())?.PlayerCombatState?.Hand.Cards;
		int num = ((cardModel == null || readOnlyList == null) ? (-1) : readOnlyList.ToList().IndexOf(cardModel));
		string fullName = action.GetType().FullName;
		uint? id = action.Id;
		return new
		{
			type = fullName,
			game_action_id = id,
			command = action.GetType().Name switch
			{
				"PlayCardAction" => "play_card", 
				"UsePotionAction" => "use_potion", 
				"EndPlayerTurnAction" => "end_turn", 
				"MoveToMapCoordAction" => "select_map_node", 
				_ => action.GetType().Name, 
			},
			net = NetAction(action),
			card = ((cardModel == null) ? null : Card(cardModel)),
			card_model_id = Value(Read(action, "CardModelId")),
			card_index = ((num >= 0) ? new int?(num) : ((int?)null)),
			target = Value(Read(action, "Target")),
			target_id = Read(action, "TargetId"),
			potion_index = Read(action, "PotionIndex"),
			destination = Value(Read(action, "_destination"))
		};
	}

	private object? NetAction(GameAction action)
	{
		if (!action.RecordableToReplay)
		{
			return new
			{
				hook_id = Read(action, "HookId")
			};
		}
		try
		{
			return Value(action.ToNetAction());
		}
		catch (Exception ex)
		{
			return new
			{
				serialization_error = ex.Message
			};
		}
	}

	public object Choice(object result)
	{
		return new
		{
			type = Read(result, "ChoiceType")?.ToString(),
			cards = new string[4] { "_canonicalCards", "_combatCards", "_deckCards", "_mutableCards" }.ToDictionary((string n) => n, (string n) => Value(Read(result, n))),
			indexes = Value(Read(result, "_indexes")),
			player_id = Read(result, "_playerId")
		};
	}

	private object[] VisibleChoices()
	{
		if (NGame.Instance == null)
		{
			return Array.Empty<object>();
		}
		List<object> list = new List<object>();
		Stack<Node> stack = new Stack<Node>();
		stack.Push(NGame.Instance);
		Node result;
		while (stack.TryPop(out result))
		{
			if (!GodotObject.IsInstanceValid(result) || (result is CanvasItem canvasItem && !canvasItem.IsVisibleInTree()))
			{
				continue;
			}
			string name = result.GetType().Name;
			if (ChoiceNodes.Contains(name) || name.EndsWith("MapPoint", StringComparison.Ordinal) || result.GetType().BaseType?.Name == "NMerchantSlot")
			{
				list.Add(new
				{
					type = name,
					node_id = Id(result),
					value = Value(result),
					enabled = (Read(result, "IsEnabled") ?? Read(Read(result, "Hitbox"), "IsEnabled")),
					map_state = Read(result, "State")?.ToString(),
					prefs = Value(Read(result, "_prefs")),
					selected = Value(Read(result, "_selectedCards") ?? Read(result, "_selectedCardHolders")),
					selecting = Read(result, "IsInCardSelection"),
					offered_cards = Value(Read(result, "_options") ?? Read(result, "_cards")),
					alternate_options = Value(Read(result, "_extraOptions"))
				});
			}
			foreach (Node child in result.GetChildren())
			{
				stack.Push(child);
			}
		}
		return list.ToArray();
	}
}
